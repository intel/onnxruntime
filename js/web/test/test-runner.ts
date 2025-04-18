// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// WebNN API currently does not have a TypeScript definition file. This file is a workaround with types generated from
// WebNN API specification.
// https://github.com/webmachinelearning/webnn/issues/677
/// <reference path="../lib/wasm/jsep/webnn/webnn.d.ts" />

import { Float16Array as Float16ArrayPolyfill } from '@petamoriken/float16';
import { expect } from 'chai';
import * as ort from 'onnxruntime-common';
import { extname } from 'path';
import { inspect } from 'util';

import { Attribute } from '../lib/onnxjs/attribute';
import { InferenceHandler, resolveBackend, SessionHandler } from '../lib/onnxjs/backend';
import { createWebGLContext } from '../lib/onnxjs/backends/webgl/webgl-context-factory';
import { Logger, Profiler } from '../lib/onnxjs/instrument';
import { Operator } from '../lib/onnxjs/operators';
import { onnx } from '../lib/onnxjs/ort-schema/protobuf/onnx';
import { Tensor } from '../lib/onnxjs/tensor';
import { ProtoUtil } from '../lib/onnxjs/util';
import { createView } from '../lib/wasm/jsep/tensor-view';
import {
  calculateTensorSizeInBytes,
  isGpuBufferSupportedType,
  isMLTensorSupportedType,
  tensorDataTypeStringToEnum,
} from '../lib/wasm/wasm-common';

import { base64toBuffer, createMockGraph, readFile } from './test-shared';
import { Test } from './test-types';

// the threshold that used to compare 2 float numbers. See above for TensorResultValidator.floatEqual().
const CPU_THRESHOLD_ABSOLUTE_ERROR = 1.0e-4;
const CPU_THRESHOLD_RELATIVE_ERROR = 1.000001;
const WEBGL_THRESHOLD_ABSOLUTE_ERROR = 1.0e-3;
const WEBGL_THRESHOLD_RELATIVE_ERROR = 1.00001;
const WEBGL_HALF_FLOAT_THRESHOLD_ABSOLUTE_ERROR = 0.1;
const WEBGL_HALF_FLOAT_THRESHOLD_RELATIVE_ERROR = 1.02;
const WEBGPU_THRESHOLD_ABSOLUTE_ERROR = 1.0e-3;
const WEBGPU_THRESHOLD_RELATIVE_ERROR = 1.00001;
const WASM_THRESHOLD_ABSOLUTE_ERROR = 1.0e-4;
const WASM_THRESHOLD_RELATIVE_ERROR = 1.000001;
const ONNXRUNTIME_THRESHOLD_ABSOLUTE_ERROR = 1.0e-3;
const ONNXRUNTIME_THRESHOLD_RELATIVE_ERROR = 1.00001;

/**
 * returns a number to represent the current timestamp in a resolution as high as possible.
 */
const now = typeof performance !== 'undefined' && performance.now ? () => performance.now() : Date.now;

function fromInternalTensor(tensor: Tensor): ort.Tensor {
  return new ort.Tensor(tensor.type, tensor.data as ort.Tensor.DataType, tensor.dims);
}

async function loadTensorProto(uriOrData: string | Uint8Array, allowInt64 = false): Promise<Test.NamedTensor> {
  const buf = typeof uriOrData === 'string' ? await readFile(uriOrData) : uriOrData;
  const tensorProto = onnx.TensorProto.decode(buf);

  let tensor: ort.Tensor;

  // by default, we don't allow (u)int64. this is for backward compatibility.
  if (
    allowInt64 &&
    tensorProto &&
    tensorProto.dataType &&
    (tensorProto.dataType === onnx.TensorProto.DataType.INT64 ||
      tensorProto.dataType === onnx.TensorProto.DataType.UINT64)
  ) {
    const signed = tensorProto.dataType === onnx.TensorProto.DataType.INT64;
    const dataConstructor = signed ? BigInt64Array : BigUint64Array;
    const length = tensorProto.rawData.byteLength / 8;
    const data = new dataConstructor(length);

    if (
      tensorProto.rawData &&
      typeof tensorProto.rawData.byteLength === 'number' &&
      tensorProto.rawData.byteLength > 0
    ) {
      const dataSource = new DataView(
        tensorProto.rawData.buffer,
        tensorProto.rawData.byteOffset,
        tensorProto.rawData.byteLength,
      );
      for (let i = 0; i < length; i++) {
        data[i] = signed ? dataSource.getBigInt64(i * 8, true) : dataSource.getBigUint64(i * 8, true);
      }
    } else {
      for (let i = 0; i < length; i++) {
        data[i] = BigInt((signed ? tensorProto.int64Data : tensorProto.uint64Data)![i].toString());
      }
    }
    tensor = new ort.Tensor(signed ? 'int64' : 'uint64', data, ProtoUtil.tensorDimsFromProto(tensorProto.dims));
  } else {
    const internalTensor = Tensor.fromProto(tensorProto);
    tensor = fromInternalTensor(internalTensor);
  }
  // add property 'name' to the tensor object.
  const namedTensor = tensor as unknown as Test.NamedTensor;
  namedTensor.name = tensorProto.name;
  return namedTensor;
}

async function loadMlProto(_uriOrData: string | Uint8Array): Promise<Test.NamedTensor> {
  return Promise.reject('not supported');
}

async function loadTensors(
  modelMetaData: { inputNames: readonly string[]; outputNames: readonly string[] },
  testCase: Test.ModelTestCase,
  backendName: string,
  fileCache?: FileCacheBuffer,
) {
  const inputs: Test.NamedTensor[] = [];
  const outputs: Test.NamedTensor[] = [];
  let dataFileType: 'none' | 'pb' | 'npy' = 'none';

  const allowInt64 = ['wasm', 'webgpu', 'webnn'].includes(backendName);

  for (const dataFile of testCase.dataFiles) {
    const ext = extname(dataFile);
    if (ext.toLowerCase() === '.pb' || ext.toLowerCase() === '.tpb') {
      if (dataFileType === 'none') {
        dataFileType = 'pb';
      }
      if (dataFileType !== 'pb') {
        throw new Error(`cannot load data from test case "${testCase.name}", multiple types of files detected`);
      }

      const uriOrData = fileCache && fileCache[dataFile] ? fileCache[dataFile] : dataFile;
      const t =
        ext.toLowerCase() === '.pb'
          ? await loadTensorProto(uriOrData, allowInt64) // onnx.TensorProto
          : await loadMlProto(uriOrData);

      const dataFileBasename = dataFile.split(/[/\\]/).pop()!;

      if (dataFileBasename.indexOf('input') !== -1) {
        inputs.push(t);
      } else if (dataFileBasename.indexOf('output') !== -1) {
        outputs.push(t);
      }
    } else {
      throw new Error(`${ext} file is not supported now`);
    }
  }

  // if model has single input/output, and tensor name is empty, we assign model's input/output names to it.
  if (modelMetaData.inputNames.length === 1 && inputs.length === 1 && !inputs[0].name) {
    inputs[0].name = modelMetaData.inputNames[0];
  }
  if (modelMetaData.outputNames.length === 1 && outputs.length === 1 && !outputs[0].name) {
    outputs[0].name = modelMetaData.outputNames[0];
  }

  testCase.inputs = inputs;
  testCase.outputs = outputs;
}

async function initializeSession(
  modelFilePath: string,
  backendHint: ort.InferenceSession.ExecutionProviderConfig,
  ioBindingMode: Test.IOBindingMode,
  profile: boolean,
  externalData: ort.InferenceSession.SessionOptions['externalData'],
  sessionOptions: ort.InferenceSession.SessionOptions,
  fileCache?: FileCacheBuffer,
): Promise<ort.InferenceSession> {
  const preloadModelData: Uint8Array | undefined =
    fileCache && fileCache[modelFilePath] ? fileCache[modelFilePath] : undefined;
  Logger.verbose(
    'TestRunner',
    `Start to load model from file: ${modelFilePath}${
      preloadModelData ? ` [preloaded(${preloadModelData.byteLength})]` : ''
    }`,
  );

  let preferredOutputLocation: ort.Tensor.DataLocation | undefined;
  if (ioBindingMode === 'gpu-location') {
    preferredOutputLocation = 'gpu-buffer';
  } else if (ioBindingMode === 'ml-location') {
    preferredOutputLocation = 'ml-tensor';
  }

  const profilerConfig = profile ? { maxNumberEvents: 65536 } : undefined;
  const sessionConfig = {
    ...sessionOptions,
    executionProviders: [backendHint],
    profiler: profilerConfig,
    enableProfiling: profile,
    preferredOutputLocation,
    externalData,
  };

  let session: ort.InferenceSession;

  try {
    if (preloadModelData) {
      session = await ort.InferenceSession.create(preloadModelData, sessionConfig);
    } else {
      const modelData = await readFile(modelFilePath);
      session = await ort.InferenceSession.create(modelData, sessionConfig);
    }
  } catch (e) {
    Logger.error(
      'TestRunner',
      `Failed to load model from file: ${modelFilePath}. ` + `Error: ${e.message} @ ${e.fileName}:${e.lineNumber}`,
    );
    throw e;
  }

  if (profile) {
    session.startProfiling();
  }

  Logger.verbose('TestRunner', `Finished loading model from file: ${modelFilePath}`);

  return session;
}

type FileCacheBuffer = {
  [filePath: string]: Uint8Array;
};
/**
 * a ModelTestContext object contains all states in a ModelTest
 */
export class ModelTestContext {
  private constructor(
    readonly session: ort.InferenceSession,
    readonly backend: string,
    readonly perfData: ModelTestContext.ModelTestPerfData,
    readonly ioBinding: Test.IOBindingMode,
    private readonly profile: boolean,
    public readonly mlContext?: MLContext,
  ) {}

  /**
   * dump the current performance data
   */
  private logPerfData() {
    const data = this.perfData;
    Logger.verbose('TestRunner.Perf', '***Perf Data Start');
    Logger.verbose('TestRunner.Perf', ` * Init          : ${data.init}`);
    Logger.verbose('TestRunner.Perf', ` * Running times : ${data.count}`);
    Logger.verbose('TestRunner.Perf', ` * FirstRun      : ${data.firstRun.toFixed(2)}`);
    const runs = data.runs;
    if (runs.length > 0) {
      Logger.verbose('TestRunner.Perf', ` * Runs          : ${runs.map((r) => r.toFixed(2)).join(', ')}`);

      if (runs.length > 1) {
        const sorted = runs.sort((a, b) => a - b);
        Logger.verbose('TestRunner.Perf', ` * Runs P50      : ${sorted[Math.floor((runs.length - 1) / 2)].toFixed(2)}`);
        const avg = runs.reduce((prev, current) => prev + current) / runs.length;
        Logger.verbose('TestRunner.Perf', ` * Runs Avg      : ${avg.toFixed(2)}`);
        const variance = runs.reduce((prev, current) => prev + (current - avg) * (current - avg));
        const sd = Math.sqrt(variance / (runs.length - 1));
        Logger.verbose('TestRunner.Perf', ` * Runs SD       : ${sd.toFixed(2)}`);
      }
    }
    Logger.verbose('TestRunner.Perf', '***Perf Data End');
  }

  async release(): Promise<void> {
    if (this.profile) {
      this.session.endProfiling();
    }
    this.logPerfData();
    await this.session.release();
  }

  /**
   * create a ModelTestContext object that used in every test cases in the given ModelTest.
   */
  static async create(
    modelTest: Test.ModelTest,
    profile: boolean,
    testOptions?: Test.Options,
  ): Promise<ModelTestContext> {
    if (this.initializing) {
      throw new Error('cannot create a ModelTestContext object when the previous creation is not done');
    }

    try {
      this.initializing = true;

      const initStart = now();
      const executionProviderConfig =
        modelTest.backend === 'webnn' ? testOptions?.webnnOptions || { name: 'webnn' } : modelTest.backend!;
      let mlContext: MLContext | undefined;
      if (['ml-tensor', 'ml-location'].includes(modelTest.ioBinding)) {
        const webnnOptions = executionProviderConfig as ort.InferenceSession.WebNNExecutionProviderOption;
        const deviceType = (webnnOptions as ort.InferenceSession.WebNNContextOptions)?.deviceType;
        const powerPreference = (webnnOptions as ort.InferenceSession.WebNNContextOptions)?.powerPreference;

        mlContext = await navigator.ml.createContext({ deviceType, powerPreference });
        (executionProviderConfig as ort.InferenceSession.WebNNExecutionProviderOption).context = mlContext;
        if (!deviceType) {
          (executionProviderConfig as ort.InferenceSession.WebNNContextOptions).deviceType = deviceType;
        }
      }
      const session = await initializeSession(
        modelTest.modelUrl,
        executionProviderConfig,
        modelTest.ioBinding,
        profile,
        modelTest.externalData,
        testOptions?.sessionOptions || {},
        this.cache,
      );

      const initEnd = now();

      for (const testCase of modelTest.cases) {
        await loadTensors(session, testCase, modelTest.backend!, this.cache);
      }

      return new ModelTestContext(
        session,
        modelTest.backend!,
        { init: initEnd - initStart, firstRun: -1, runs: [], count: 0 },
        modelTest.ioBinding,
        profile,
        mlContext,
      );
    } finally {
      this.initializing = false;
    }
  }

  /**
   * set the global file cache for looking up model and tensor protobuf files.
   */
  static setCache(cache: Test.FileCache): void {
    const keys = Object.keys(cache);
    Logger.info('TestRunner', `Setting up file cache... Entry count: ${keys.length}.`);
    for (const key of keys) {
      this.cache[key] = base64toBuffer(cache[key]);
    }
  }

  private static initializing = false;
  private static cache: FileCacheBuffer = {};
}

export declare namespace ModelTestContext {
  export interface ModelTestPerfData {
    init: number;
    firstRun: number;
    runs: number[];
    count: number;
  }
}

export class TensorResultValidator {
  private readonly absoluteThreshold: number;
  private readonly relativeThreshold: number;
  private readonly maxFloatValue: number = 3.4028234663852886e38;

  private static isHalfFloat: boolean | undefined;

  constructor(backend: string) {
    if (backend === 'cpu') {
      this.absoluteThreshold = CPU_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = CPU_THRESHOLD_RELATIVE_ERROR;
    } else if (backend === 'webgl') {
      if (TensorResultValidator.isHalfFloat === undefined) {
        TensorResultValidator.isHalfFloat = !createWebGLContext(ort.env.webgl.contextId).isRenderFloat32Supported;
      }
      if (TensorResultValidator.isHalfFloat) {
        this.maxFloatValue = 65504;
        this.absoluteThreshold = WEBGL_HALF_FLOAT_THRESHOLD_ABSOLUTE_ERROR;
        this.relativeThreshold = WEBGL_HALF_FLOAT_THRESHOLD_RELATIVE_ERROR;
      } else {
        this.absoluteThreshold = WEBGL_THRESHOLD_ABSOLUTE_ERROR;
        this.relativeThreshold = WEBGL_THRESHOLD_RELATIVE_ERROR;
      }
    } else if (backend === 'webgpu') {
      this.absoluteThreshold = WEBGPU_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = WEBGPU_THRESHOLD_RELATIVE_ERROR;
    } else if (backend === 'wasm' || backend === 'webnn') {
      this.absoluteThreshold = WASM_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = WASM_THRESHOLD_RELATIVE_ERROR;
    } else if (backend === 'onnxruntime') {
      this.absoluteThreshold = ONNXRUNTIME_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = ONNXRUNTIME_THRESHOLD_RELATIVE_ERROR;
    } else {
      throw new Error(`backend not supported: ${backend}`);
    }
  }

  checkTensorResult(actual: Tensor[], expected: Tensor[]): void {
    this.checkApiTensorResult(actual.map(fromInternalTensor), expected.map(fromInternalTensor));
  }

  checkApiTensorResult(actual: ort.Tensor[], expected: ort.Tensor[]): void {
    // check output size
    expect(actual.length, 'size of output tensors').to.equal(expected.length);

    // compare output one-by-one
    for (let i = 0; i < actual.length; ++i) {
      const match = this.areEqual(actual[i], expected[i]);
      if (!match) {
        Logger.error(
          'TestRunner',
          `Tensor mismatch: \nACTUAL: type=${actual[i].type}; dims=[${actual[i].dims}]; data=[${actual[i].data}]\nEXPECT: type=${expected[i].type}; dims=[${expected[i].dims}]; data=[${expected[i].data}]`,
        );
      }
      expect(match, 'tensor data should match').to.be.true;
    }
  }

  checkNamedTensorResult(actual: Record<string, ort.Tensor>, expected: Test.NamedTensor[]): void {
    // check output size
    expect(Object.getOwnPropertyNames(actual).length, 'size of output tensors').to.equal(expected.length);

    // check output mapping
    for (const expectedOneOutput of expected) {
      expect(actual, 'keys of output tensors').to.contain.keys(expectedOneOutput.name);
    }

    this.checkApiTensorResult(
      expected.map((i) => actual[i.name]!),
      expected,
    );
  }

  // This function check whether 2 tensors should be considered as 'match' or not
  areEqual(actual: ort.Tensor, expected: ort.Tensor): boolean {
    if (!actual || !expected) {
      return false;
    }
    if (!actual.dims || !expected.dims) {
      return false;
    }

    const actualDims = actual.dims;
    const actualType = actual.type;
    const expectedDims = expected.dims;
    const expectedType = expected.type;

    if (actualType !== expectedType) {
      return false;
    }
    if (actualDims.length !== expectedDims.length) {
      return false;
    }

    for (let i = 0; i < actualDims.length; i++) {
      if (actualDims[i] !== expectedDims[i]) {
        return false;
      }
    }

    switch (actualType) {
      case 'string':
        return this.strictEqual(actual.data, expected.data);

      case 'float16': {
        const actualData = actual.data as Uint16Array;
        const actualDataBuffer = actualData.buffer;
        const actualDataByteOffset = actualData.byteOffset;
        const actualDataLength = actualData.length;
        const actualDataFloat32Array = new Float32Array(
          new Float16ArrayPolyfill(actualDataBuffer, actualDataByteOffset, actualDataLength),
        );

        const expectedData = expected.data as Uint16Array;
        const expectedDataBuffer = expectedData.buffer;
        const expectedDataByteOffset = expectedData.byteOffset;
        const expectedDataLength = expectedData.length;
        const expectedDataFloat32Array = new Float32Array(
          new Float16ArrayPolyfill(expectedDataBuffer, expectedDataByteOffset, expectedDataLength),
        );

        return this.floatEqual(actualDataFloat32Array, expectedDataFloat32Array);
      }

      case 'float32':
      case 'float64':
        return this.floatEqual(
          actual.data as number[] | Float32Array | Float64Array,
          expected.data as number[] | Float32Array | Float64Array,
        );

      case 'uint8':
      case 'int8':
      case 'uint16':
      case 'int16':
      case 'int32':
      case 'uint32':
      case 'int64':
      case 'bool':
      case 'int4':
      case 'uint4':
        return TensorResultValidator.integerEqual(
          actual.data as number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array,
          expected.data as number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array,
        );

      default:
        throw new Error('type not implemented or not supported');
    }
  }
  strictEqual<T>(actual: T, expected: T): boolean {
    try {
      expect(actual).to.deep.equal(expected);
      return true;
    } catch {
      return false;
    }
  }
  floatEqual(
    actual: number[] | Float32Array | Float64Array,
    expected: number[] | Float32Array | Float64Array,
  ): boolean {
    if (actual.length !== expected.length) {
      return false;
    }

    for (let i = actual.length - 1; i >= 0; i--) {
      const a = actual[i];
      let b = expected[i];

      if (a === b) {
        continue; // exact the same value, treat as equal
      }

      // check for NaN
      //
      if (Number.isNaN(a) && Number.isNaN(b)) {
        continue; // 2 numbers are NaN, treat as equal
      }
      if (Number.isNaN(a) || Number.isNaN(b)) {
        Logger.error('Validator', `a or b isNan -- index:${i}: actual=${actual[i]},expected=${expected[i]}`);
        return false; // one is NaN and the other is not
      }

      // check for Infinity
      //
      if (!Number.isFinite(a) || !Number.isFinite(b)) {
        Logger.error('Validator', `a or b is Infinity -- index:${i}: actual=${actual[i]},expected=${expected[i]}`);
        return false; // at least one is Infinity and the other is not or their sign is different
      }

      // normalize value of b
      b = Math.max(Math.min(expected[i], this.maxFloatValue), -this.maxFloatValue);

      // Comparing 2 float numbers: (Suppose a >= b)
      //
      // if ( a - b < ABSOLUTE_ERROR || 1.0 < a / b < RELATIVE_ERROR)
      //   test pass
      // else
      //   test fail
      // endif
      //
      if (Math.abs(actual[i] - expected[i]) < this.absoluteThreshold) {
        continue; // absolute error check pass
      }
      if (a !== 0 && b !== 0 && a / b < this.relativeThreshold && b / a < this.relativeThreshold) {
        continue; // relative error check pass
      }

      // if code goes here, it means both (abs/rel) check failed.
      Logger.error('Validator', `abs/rel check failed-- index:${i}: actual=${actual[i]},expected=${expected[i]}`);
      return false;
    }

    return true;
  }
  static integerEqual(
    actual: number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array,
    expected: number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array,
  ): boolean {
    if (actual.length !== expected.length) {
      return false;
    }

    for (let i = actual.length - 1; i >= 0; i--) {
      if (actual[i] !== expected[i]) {
        return false;
      }
    }

    return true;
  }
}

async function createGpuTensorForInput(cpuTensor: ort.Tensor): Promise<ort.Tensor> {
  if (!isGpuBufferSupportedType(cpuTensor.type) || Array.isArray(cpuTensor.data)) {
    throw new Error(`createGpuTensorForInput can not work with ${cpuTensor.type} tensor`);
  }
  const device = await ort.env.webgpu.device;
  const gpuBuffer = device.createBuffer({
    // eslint-disable-next-line no-bitwise
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    size: Math.ceil(cpuTensor.data.byteLength / 16) * 16,
    mappedAtCreation: true,
  });
  const arrayBuffer = gpuBuffer.getMappedRange();
  new Uint8Array(arrayBuffer).set(
    new Uint8Array(cpuTensor.data.buffer, cpuTensor.data.byteOffset, cpuTensor.data.byteLength),
  );
  gpuBuffer.unmap();

  // TODO: how to "await" for the copy to finish, so that we can get more accurate performance data?

  return ort.Tensor.fromGpuBuffer(gpuBuffer, {
    dataType: cpuTensor.type,
    dims: cpuTensor.dims,
    dispose: () => gpuBuffer.destroy(),
  });
}

async function createGpuTensorForOutput(type: ort.Tensor.Type, dims: readonly number[]) {
  if (!isGpuBufferSupportedType(type)) {
    throw new Error(`createGpuTensorForOutput can not work with ${type} tensor`);
  }

  const size = calculateTensorSizeInBytes(tensorDataTypeStringToEnum(type), dims)!;

  const device = await ort.env.webgpu.device;
  const gpuBuffer = device.createBuffer({
    // eslint-disable-next-line no-bitwise
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    size: Math.ceil(size / 16) * 16,
  });

  return ort.Tensor.fromGpuBuffer(gpuBuffer, {
    dataType: type,
    dims,
    dispose: () => gpuBuffer.destroy(),
    download: async () => {
      const stagingBuffer = device.createBuffer({
        // eslint-disable-next-line no-bitwise
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        size: gpuBuffer.size,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(gpuBuffer, 0, stagingBuffer, 0, gpuBuffer.size);
      device.queue.submit([encoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const arrayBuffer = stagingBuffer.getMappedRange().slice(0, size);
      stagingBuffer.destroy();

      return createView(arrayBuffer, type) as ort.Tensor.DataTypeMap[ort.Tensor.GpuBufferDataTypes];
    },
  });
}

async function createMLTensorForOutput(mlContext: MLContext, type: ort.Tensor.Type, dims: readonly number[]) {
  if (!isMLTensorSupportedType(type)) {
    throw new Error(`createMLTensorForOutput can not work with ${type} tensor`);
  }

  const dataType = type === 'bool' ? 'uint8' : type;

  const mlTensor = await mlContext.createTensor({
    dataType,
    shape: dims as number[],
    // Assign both shape and dimensions while transitioning to new API.
    dimensions: dims as number[],
    usage: typeof MLTensorUsage == 'undefined' ? undefined : MLTensorUsage.READ,
    readable: true,
  });

  return ort.Tensor.fromMLTensor(mlTensor, {
    dataType: type,
    dims,
    dispose: () => mlTensor.destroy(),
    download: async () => {
      const arrayBuffer = await mlContext.readTensor(mlTensor);
      return createView(arrayBuffer, type) as ort.Tensor.DataTypeMap[ort.Tensor.MLTensorDataTypes];
    },
  });
}

async function createMLTensorForInput(mlContext: MLContext, cpuTensor: ort.Tensor): Promise<ort.Tensor> {
  if (!isMLTensorSupportedType(cpuTensor.type) || Array.isArray(cpuTensor.data)) {
    throw new Error(`createMLTensorForInput can not work with ${cpuTensor.type} tensor`);
  }
  const dataType = cpuTensor.type === 'bool' ? 'uint8' : cpuTensor.type;
  const mlTensor = await mlContext.createTensor({
    dataType,
    shape: cpuTensor.dims as number[],
    // Assign both shape and dimensions while transitioning to new API.
    dimensions: cpuTensor.dims as number[],
    usage: typeof MLTensorUsage == 'undefined' ? undefined : MLTensorUsage.WRITE,
    writable: true,
  });
  mlContext.writeTensor(mlTensor, cpuTensor.data);
  return ort.Tensor.fromMLTensor(mlTensor, {
    dataType: cpuTensor.type,
    dims: cpuTensor.dims,
    dispose: () => mlTensor.destroy(),
  });
}

export async function sessionRun(options: {
  session: ort.InferenceSession;
  feeds: Record<string, ort.Tensor>;
  outputsMetaInfo: Record<string, Pick<ort.Tensor, 'dims' | 'type'>>;
  ioBinding: Test.IOBindingMode;
  mlContext?: MLContext;
}): Promise<[number, number, ort.InferenceSession.OnnxValueMapType]> {
  const session = options.session;
  const feeds = options.feeds;
  const fetches: Record<string, ort.Tensor> = {};

  // currently we only support IO Binding for WebGPU and WebNN
  //
  // For inputs, we create tensors on 'gpu-tensor', 'gpu-location', 'ml-tensor', and 'ml-location' binding testing
  // modes.
  // For outputs, we create tensors on 'gpu-tensor' and 'ml-tensor' binding testing modes.
  //              in 'gpu-device' binding mode, outputs are not pre-allocated.
  const shouldUploadInput = ['gpu-tensor', 'gpu-location', 'ml-location', 'ml-tensor'].includes(options.ioBinding);
  const shouldUploadOutput = options.ioBinding === 'gpu-tensor' || options.ioBinding === 'ml-tensor';
  try {
    if (shouldUploadInput) {
      // replace the CPU tensors in feeds into GPU tensors
      for (const name in feeds) {
        if (Object.hasOwnProperty.call(feeds, name)) {
          if (feeds[name].size > 0) {
            if (options.ioBinding === 'ml-location' || options.ioBinding === 'ml-tensor') {
              feeds[name] = await createMLTensorForInput(options.mlContext!, feeds[name]);
            } else {
              feeds[name] = await createGpuTensorForInput(feeds[name]);
            }
          }
        }
      }
    }

    if (shouldUploadOutput) {
      for (const name in options.outputsMetaInfo) {
        if (Object.hasOwnProperty.call(options.outputsMetaInfo, name)) {
          const { type, dims } = options.outputsMetaInfo[name];
          if (dims.some((d) => d === 0)) {
            fetches[name] = new ort.Tensor(type, [], dims);
          } else {
            if (options.ioBinding === 'ml-tensor') {
              fetches[name] = await createMLTensorForOutput(options.mlContext!, type, dims);
            } else {
              fetches[name] = await createGpuTensorForOutput(type, dims);
            }
          }
        }
      }
    }

    const start = now();
    Logger.verbose('TestRunner', `Timestamp before session run: ${start}`);
    const outputs = await (shouldUploadOutput
      ? session.run(feeds, fetches)
      : session.run(feeds, Object.getOwnPropertyNames(options.outputsMetaInfo)));
    const end = now();
    Logger.verbose('TestRunner', `Timestamp after session run: ${end}`);

    // download each output tensor if needed
    for (const name in outputs) {
      if (Object.hasOwnProperty.call(outputs, name)) {
        const tensor = outputs[name];
        // Tensor.getData(true) release the underlying resource
        await tensor.getData(true);
      }
    }

    return [start, end, outputs];
  } finally {
    // dispose the GPU tensors in feeds
    for (const name in feeds) {
      if (Object.hasOwnProperty.call(feeds, name)) {
        const tensor = feeds[name];
        tensor.dispose();
      }
    }
  }
}

/**
 * run a single model test case. the inputs/outputs tensors should already been prepared.
 */
export async function runModelTestSet(
  context: ModelTestContext,
  testCase: Test.ModelTestCase,
  testName: string,
): Promise<void> {
  Logger.verbose('TestRunner', `Start to run test data from folder: ${testName}/${testCase.name}`);
  Logger.verbose('TestRunner', `Start to run test data from folder: ${testCase.name}`);
  const validator = new TensorResultValidator(context.backend);
  try {
    const feeds: Record<string, ort.Tensor> = {};
    const outputsMetaInfo: Record<string, ort.Tensor> = {};
    testCase.inputs!.forEach((tensor) => (feeds[tensor.name] = tensor));
    testCase.outputs!.forEach((tensor) => (outputsMetaInfo[tensor.name] = tensor));
    const [start, end, outputs] = await sessionRun({
      session: context.session,
      feeds,
      outputsMetaInfo,
      ioBinding: context.ioBinding,
      mlContext: context.mlContext,
    });
    if (context.perfData.count === 0) {
      context.perfData.firstRun = end - start;
    } else {
      context.perfData.runs.push(end - start);
    }
    context.perfData.count++;

    Logger.verbose('TestRunner', `Finished running model from file: ${testCase.name}`);
    Logger.verbose('TestRunner', ' Stats:');
    Logger.verbose('TestRunner', `  Input(s): ${testCase.inputs!.length}`);
    testCase.inputs!.forEach((i) => {
      Logger.verbose('TestRunner', `   '${i.name}': ${i.type}[${i.dims.join(',')}]`);
    });
    Logger.verbose('TestRunner', `  Output(s): ${Object.keys(outputs).length}`);
    for (const name in outputs) {
      if (Object.hasOwnProperty.call(outputs, name)) {
        const tensor = outputs[name];
        Logger.verbose('TestRunner', `   '${name}': ${tensor.type}[${tensor.dims.join(',')}]`);
      }
    }

    validator.checkNamedTensorResult(outputs, testCase.outputs!);

    Logger.verbose('TestRunner', '  Result: PASS');
  } catch (e) {
    Logger.error('TestRunner', '  Result: FAILED');
    Logger.error('TestRunner', `Failed to run test data from folder: ${testCase.name}. Error: ${inspect(e)}`);
    throw e;
  }
}

function initializeOperator(
  sessionHandler: SessionHandler,
  opType: string,
  attributeValues: readonly Test.AttributeValue[],
  opsetImports: readonly Test.OperatorTestOpsetImport[],
): Operator {
  const attributes = new Attribute(undefined);
  attributeValues.forEach((value) => attributes.set(value.name, value.type, value.data));
  const graph = createMockGraph(opType, attributes);
  return sessionHandler.resolve(graph.getNodes()[0], opsetImports, graph);
}

/**
 * a OpTestContext object contains all states in a OpTest. used for webgl backend.
 */
export class OpTestContext {
  static profiler = Profiler.create();

  readonly backendHint: string;
  sessionHandler: SessionHandler;
  inferenceHandler: InferenceHandler;

  constructor(protected opTest: Test.OperatorTest) {
    this.backendHint = opTest.backend ?? 'cpu';
  }
  createOperator(): Operator {
    return initializeOperator(this.sessionHandler, this.opTest.operator, this.opTest.attributes || [], [
      this.opTest.opset ?? { domain: '', version: 7 },
    ]);
  }

  async dispose(): Promise<void> {
    this.inferenceHandler.dispose();
    this.sessionHandler.dispose();
  }

  async init(): Promise<void> {
    const backend = await resolveBackend(this.backendHint);
    this.sessionHandler = backend.createSessionHandler({ profiler: OpTestContext.profiler });
    this.inferenceHandler = this.sessionHandler.createInferenceHandler();
  }
}

/**
 * a ProtoOpTestContext uses a protobuf model for operator test. used for ORT based backend.
 */
export class ProtoOpTestContext {
  private readonly loadedData: Uint8Array; // model data, inputs, outputs
  session: ort.InferenceSession;
  readonly backendHint: string;
  readonly ioBindingMode: Test.IOBindingMode;
  constructor(
    test: Test.OperatorTest,
    private readonly downloadModel: boolean,
    private readonly sessionOptions: ort.InferenceSession.SessionOptions = {},
  ) {
    const opsetImport = onnx.OperatorSetIdProto.create(test.opset);
    const operator = test.operator;
    const attribute = (test.attributes || []).map((attr) => {
      const protoAttr = onnx.AttributeProto.create({ name: attr.name });
      switch (attr.type) {
        case 'float':
          protoAttr.type = onnx.AttributeProto.AttributeType.FLOAT;
          protoAttr.f = attr.data as number;
          break;
        case 'int':
          protoAttr.type = onnx.AttributeProto.AttributeType.INT;
          protoAttr.i = attr.data as number;
          break;
        case 'string':
          protoAttr.type = onnx.AttributeProto.AttributeType.STRING;
          protoAttr.s = new TextEncoder().encode(attr.data as string);
          break;
        case 'floats':
          protoAttr.type = onnx.AttributeProto.AttributeType.FLOATS;
          protoAttr.floats = attr.data as number[];
          break;
        case 'ints':
          protoAttr.type = onnx.AttributeProto.AttributeType.INTS;
          protoAttr.ints = attr.data as number[];
          break;
        case 'strings':
          protoAttr.type = onnx.AttributeProto.AttributeType.STRINGS;
          protoAttr.strings = (attr.data as string[]).map((s) => new TextEncoder().encode(s));
          break;
        default:
          throw new Error(`Unsupported attribute type: ${attr.type}`);
      }
      return protoAttr;
    });

    if (test.cases.length === 0) {
      throw new Error(`No test cases found for test: ${test.name} [${test.operator}]`);
    }
    const inputCount = test.cases[0].inputs!.length;
    const outputCount = test.cases[0].outputs!.length;
    if (
      test.cases.some((testCase) => testCase.inputs!.length !== inputCount || testCase.outputs!.length !== outputCount)
    ) {
      throw new Error(
        `Test cases for test: ${test.name} [${test.operator}] must have the same number of inputs and outputs`,
      );
    }
    const inputsOmitted = test.cases[0].inputs.map((input) => !input.data);
    const outputsOmitted = test.cases[0].outputs.map((output) => !output.data);
    for (let caseIndex = 1; caseIndex < test.cases.length; caseIndex++) {
      const testCase = test.cases[caseIndex];
      for (let i = 0; i < inputCount; i++) {
        if (inputsOmitted[i] !== !testCase.inputs![i].data) {
          throw new Error(
            `Test cases for test: ${test.name} [${test.operator}] must have consistent inputs data availability. Data of input[${i}] in testCase #0 and #${caseIndex} should be both available or both omitted.`,
          );
        }
      }
      for (let i = 0; i < outputCount; i++) {
        if (outputsOmitted[i] !== !testCase.outputs![i].data) {
          throw new Error(
            `Test cases for test: ${test.name} [${test.operator}] must have consistent outputs data availability. Data of output[${i}] in testCase #0 and #${caseIndex} should be both available or both omitted.`,
          );
        }
      }
    }

    const model = onnx.ModelProto.create();
    model.irVersion = onnx.Version.IR_VERSION;
    model.opsetImport.push(opsetImport);
    model.graph = onnx.GraphProto.create();

    model.graph.node = [
      onnx.NodeProto.create({
        input: test.cases[0].inputs!.map((t, i) => (t.data ? `input_${i}` : '')),
        output: test.cases[0].outputs!.map((t, i) => (t.data ? `output_${i}` : '')),
        opType: operator,
        domain: test.opset?.domain,
        name: operator,
        attribute,
      }),
    ];

    // normalize input shape definitions
    let normalizedInputShapeDefinitions: ReadonlyArray<Test.InputShapeDefinition | undefined>;
    if (!test.inputShapeDefinitions || test.inputShapeDefinitions === 'none') {
      // if inputShapeDefinitions is not specified, use undefined for all inputs
      normalizedInputShapeDefinitions = new Array(inputCount).fill(undefined);
    } else if (test.inputShapeDefinitions === 'rankOnly') {
      // check if all test cases have data
      if (test.cases.some((testCase) => testCase.inputs!.some((input) => !input.data || !input.dims))) {
        throw new Error(
          `Test cases for test: ${test.name} [${test.operator}] must have data for each inputs when inputShapeDefinitions is 'rankOnly'`,
        );
      }

      // if inputShapeDefinitions is 'rankOnly', use semantic names for all inputs. This means only rank is specified.
      normalizedInputShapeDefinitions = test.cases[0].inputs!.map((input: Test.TensorValue, i) =>
        input.dims.map((_, j) => `_input_${i}_d${j}`),
      );

      // check if all test cases have the same rank for each inputs
      if (
        test.cases.some((testCase) =>
          testCase.inputs!.some(
            (input: Test.TensorValue, i) =>
              input.dims.length !== (test.cases[0].inputs![i] as Test.TensorValue).dims.length,
          ),
        )
      ) {
        throw new Error(
          `Test cases for test: ${test.name} [${test.operator}] must have the same rank for each inputs in different test cases`,
        );
      }
    } else if (test.inputShapeDefinitions === 'static') {
      // check if all test cases have data
      if (test.cases.some((testCase) => testCase.inputs!.some((input) => !input.data || !input.dims))) {
        throw new Error(
          `Test cases for test: ${test.name} [${test.operator}] must have data for each inputs when inputShapeDefinitions is 'rankOnly'`,
        );
      }

      // if inputShapeDefinitions is 'static', use the shape of the first test case for all inputs.
      normalizedInputShapeDefinitions = test.cases[0].inputs!.map((input: Test.TensorValue) => input.dims);

      // check if all test cases have the same shape for each inputs
      if (
        test.cases.some((testCase) =>
          testCase.inputs!.some(
            (input: Test.TensorValue, i) =>
              !TensorResultValidator.integerEqual(input.dims, (test.cases[0].inputs![i] as Test.TensorValue).dims),
          ),
        )
      ) {
        throw new Error(
          `Test cases for test: ${test.name} [${test.operator}] must have the same shape for each inputs in different test cases`,
        );
      }
    } else {
      // if inputShapeDefinitions is specified as an array, use it as is.
      // check if inputShapeDefinitions has the same number of inputs as test cases
      if (test.inputShapeDefinitions && test.inputShapeDefinitions.length !== inputCount) {
        throw new Error(
          `Input shape definitions for test: ${test.name} [${test.operator}] must have the same number of inputs`,
        );
      }
      normalizedInputShapeDefinitions = test.inputShapeDefinitions;
    }

    model.graph.input = test.cases[0]
      .inputs!.map((input, i) => {
        const shapeDefinition = normalizedInputShapeDefinitions[i];
        const shape = shapeDefinition
          ? onnx.TensorShapeProto.create({
              dim: shapeDefinition.map((dim) =>
                onnx.TensorShapeProto.Dimension.create(typeof dim === 'string' ? { dimParam: dim } : { dimValue: dim }),
              ),
            })
          : undefined;
        return onnx.ValueInfoProto.create({
          name: `input_${i}`,
          type: onnx.TypeProto.create({
            tensorType: onnx.TypeProto.Tensor.create({ elemType: tensorDataTypeStringToEnum(input.type), shape }),
          }),
        });
      })
      .filter((_, i) => test.cases[0].inputs![i].data);

    model.graph.output = test.cases[0]
      .outputs!.map((output, i) =>
        onnx.ValueInfoProto.create({
          name: `output_${i}`,
          type: onnx.TypeProto.create({
            tensorType: onnx.TypeProto.Tensor.create({ elemType: tensorDataTypeStringToEnum(output.type) }),
          }),
        }),
      )
      .filter((_, i) => test.cases[0].outputs![i].data);

    model.graph.name = test.name;

    this.backendHint = test.backend!;
    this.ioBindingMode = test.ioBinding;
    this.loadedData = onnx.ModelProto.encode(model).finish().slice();

    if (this.downloadModel) {
      const modelFile = new File([this.loadedData], `op_test_generated_model_${test.name}.onnx`, {
        type: 'application/octet-stream',
      });
      const modelTempUrl = URL.createObjectURL(modelFile);
      const a = document.createElement('a');
      a.href = modelTempUrl;
      a.download = modelFile.name;
      a.target = '_blank';
      a.click();
      URL.revokeObjectURL(modelTempUrl);
    }
  }
  async init(): Promise<void> {
    this.session = await ort.InferenceSession.create(this.loadedData, {
      executionProviders: [this.backendHint],
      preferredOutputLocation: this.ioBindingMode === 'gpu-location' ? ('gpu-buffer' as const) : undefined,
      ...this.sessionOptions,
    });
  }

  async dispose(): Promise<void> {
    await this.session.release();
  }
}

async function runProtoOpTestcase(
  session: ort.InferenceSession,
  testCase: Test.OperatorTestCase,
  ioBindingMode: Test.IOBindingMode,
  validator: TensorResultValidator,
): Promise<void> {
  const feeds: Record<string, ort.Tensor> = {};
  const fetches: Record<string, Pick<ort.Tensor, 'dims' | 'type'>> = {};

  const createTensor = (type: ort.Tensor.Type, data: number[], dims: readonly number[]): ort.Tensor => {
    let buffer: number[] | BigUint64Array | BigInt64Array | Uint16Array | Uint8Array = data;
    if (type === 'uint64') {
      buffer = BigUint64Array.from(data.map(BigInt));
    } else if (type === 'int64') {
      buffer = BigInt64Array.from(data.map(BigInt));
    } else if (type === 'float16') {
      const dataArr = Float16ArrayPolyfill.from(data);
      buffer = new Uint16Array(dataArr.buffer, dataArr.byteOffset, dataArr.byteLength / 2);
    } else if (type === 'uint4' || type === 'int4') {
      buffer = new Uint8Array(calculateTensorSizeInBytes(tensorDataTypeStringToEnum(type), dims)!);
      // encode (u)int4 data into Uint8Array
      for (let j = 0; j < data.length; j++) {
        /* eslint-disable no-bitwise */
        const byteIndex = j >> 1;
        const bitOffset = (j & 1) << 2;
        buffer[byteIndex] |= data[j] << bitOffset;
        /* eslint-enable no-bitwise */
      }
    }
    return new ort.Tensor(type, buffer, dims);
  };

  testCase.inputs.forEach((input, i) => {
    if (input.data) {
      feeds[`input_${i}`] = createTensor(input.type, input.data, input.dims);
    }
  });

  const outputs: ort.Tensor[] = [];
  const expectedOutputNames: string[] = [];
  testCase.outputs.forEach((output, i) => {
    if (output.data) {
      outputs.push(createTensor(output.type, output.data, output.dims));
      expectedOutputNames.push(`output_${i}`);
      fetches[`output_${i}`] = { dims: output.dims, type: output.type };
    }
  });

  const [, , results] = await sessionRun({ session, feeds, outputsMetaInfo: fetches, ioBinding: ioBindingMode });

  const actualOutputNames = Object.getOwnPropertyNames(results);
  expect(actualOutputNames.length).to.equal(expectedOutputNames.length);
  expect(actualOutputNames).to.have.members(expectedOutputNames);

  const actualOutputs = actualOutputNames.map((name) => results[name]);
  validator.checkApiTensorResult(actualOutputs, outputs);
}

function createTensor(dims: number[], type: Tensor.DataType, data: number[]): Tensor {
  const tensor = new Tensor(dims, type);
  for (let i = 0; i < data.length; ++i) {
    tensor.data[i] = data[i];
  }
  return tensor;
}

async function runOpTestcase(
  inferenceHandler: InferenceHandler,
  operator: Operator,
  testcase: Test.OperatorTestCase,
  validator: TensorResultValidator,
): Promise<void> {
  testcase.inputs.forEach((input: Test.TensorValue, i) => {
    Logger.verbose('TestOpRunner', `   Input '${i}': ${input.type}[${input.dims.join(',')}]`);
  });
  const inputTensors = testcase.inputs.map((input: Test.TensorValue) =>
    createTensor(input.dims, input.type as Tensor.DataType, input.data),
  );

  const results = operator.impl(inferenceHandler, inputTensors, operator.context);

  // try async data read.
  for (const result of results) {
    try {
      await result.getData();
    } catch {}
  }

  results.forEach((output, i) => {
    Logger.verbose('TestOpRunner', `  Result'${i}': ${output.type}[${output.dims.join(',')}]`);
  });
  const expectedTensors = testcase.outputs.map((output: Test.TensorValue) =>
    createTensor(output.dims, output.type as Tensor.DataType, output.data),
  );
  validator.checkTensorResult(results, expectedTensors);
}

/**
 * run a single operator test case.
 */
export async function runOpTest(
  testcase: Test.OperatorTestCase,
  context: ProtoOpTestContext | OpTestContext,
): Promise<void> {
  if (context instanceof ProtoOpTestContext) {
    await runProtoOpTestcase(
      context.session,
      testcase,
      context.ioBindingMode,
      new TensorResultValidator(context.backendHint),
    );
  } else {
    await runOpTestcase(
      context.inferenceHandler,
      context.createOperator(),
      testcase,
      new TensorResultValidator(context.backendHint),
    );
  }
}
