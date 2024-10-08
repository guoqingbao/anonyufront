// Copyright 2023, ********************
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
use half::f16;
use num_enum::TryFromPrimitive;
use pyo3::prelude::*;
// use pyo3::prelude::{pymodule, PyModule, PyResult};
use pyo3::pyclass;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum DataType {
    Int32 = 100,
    Int64,
    Half,  //fp16
    BHalf, //bf16
    Float,
    Double,
    Bool,
}

// impl PartialEq for DataType {
//     fn eq(&self, other: &Self) -> bool {
//         match self {
//             DataType::Half(va) => match other {
//                 DataType::Half(vb) => {(va.to_f32() - vb.to_f32()).abs() < 0.001},
//                 _ => panic!("Invalid data type!"),
//             },
//             DataType::Float(va) => match other {
//                 DataType::Float(vb) => {(va - vb).abs() < 0.001},
//                 _ => panic!("Invalid data type!"),
//             },
//             DataType::Double(va) => match other {
//                 DataType::Double(vb) => {(va - vb).abs() < 0.001},
//                 _ => panic!("Invalid data type!"),
//             },
//             _ => panic!()

//         }

//     }
// }

// impl Eq for DataType {}

// impl Hash for DataType {
//     fn hash<H>(&self, state: &mut H) where H: Hasher {
//         state.write_u8(4);
//     }
// }

#[pymethods]
impl DataType {
    pub fn bytes(&self) -> usize {
        match self {
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Half => 2,
            DataType::BHalf => 2,
            DataType::Float => 4,
            DataType::Double => 8,
            DataType::Bool => 2,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceType {
    CPU,
    GPU,
    GCU,
}

impl DeviceType {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceType::CPU => "cpu",
            DeviceType::GPU => "gpu",
            DeviceType::GCU => "gcu",
            // _ => panic!("Invalid device type!"),
        }
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum OpType {
    CONV2D = 2011,
    EMBEDDING,
    POOL2D,
    LINEAR,
    SOFTMAX,
    CONCAT,
    FLAT,
    MSELOSS,
    BATCH_NORM,
    RELU,
    SIGMOID,
    TANH,
    ELU,
    DROPOUT,
    BATCH_MATMUL,
    SPLIT,
    RESHAPE,
    TRANSPOSE,
    REVERSE,
    EXP,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    POW,
    MEAN,
    SQRT,
    RSQRT,
    RECIPROCAL,
    NEG,
    SIN,
    COS,
    INPUT,
    OUTPUT,
    MULTIHEAD_ATTENTION,
    GETITEM,
    GETATTR,
    EXPAND,
    LAYER_NORM,
    FLOOR_DIVIDE,
    IDENTITY,
    GELU,
    PERMUTE,
    SCALAR_MULTIPLY,
    SCALAR_FLOORDIV,
    SCALAR_ADD,
    SCALAR_SUB,
    SCALAR_TRUEDIV,
    PARAMETER,
    FLOAT,
    CONTIGUOUS,
    TO,
    UNSQUEEZE,
    TYPE_AS,
    VIEW,
    ATTRIBUTE,
    EQ,
    SLICE,
    CALL,
    ASSERT,
    CHUNK,
    HARDSWISH,
    HARDSIGMOID,
    SILU,
    MASKEDFILL,
    REPEAT,
    UNIFORM_LIKE,
    LESS,
    CAST,
    TENSOR,
    MATMUL,
    CLIP,
    ERF,
    BOOL,
    INVERT,
    AND,
    DETACH,
    CUMSUM,
    ARANGE,
}

#[pymethods]
impl OpType {
    pub fn as_str(&self) -> &'static str {
        match self {
            OpType::CONV2D => "conv2d",
            OpType::EMBEDDING => "embedding",
            OpType::POOL2D => "pool2d",
            OpType::LINEAR => "linear",
            OpType::SOFTMAX => "softmax",
            OpType::CONCAT => "concat",
            OpType::FLAT => "flat",
            OpType::MSELOSS => "mseloss",
            OpType::BATCH_NORM => "batchnorm",
            OpType::RELU => "relu",
            OpType::SIGMOID => "sigmoid",
            OpType::TANH => "tanh",
            OpType::ELU => "elu",
            OpType::DROPOUT => "dropout",
            OpType::BATCH_MATMUL => "batch_matmul",
            OpType::SPLIT => "split",
            OpType::RESHAPE => "reshape",
            OpType::TRANSPOSE => "transpose",
            OpType::REVERSE => "reverse",
            OpType::EXP => "exp",
            OpType::ADD => "add",
            OpType::SUBTRACT => "subtract",
            OpType::MULTIPLY => "multiply",
            OpType::DIVIDE => "divide",
            OpType::POW => "pow",
            OpType::MEAN => "mean",
            OpType::SQRT => "sqrt",
            OpType::RSQRT => "rsqrt",
            OpType::RECIPROCAL => "reciprocal",
            OpType::NEG => "neg",
            OpType::SIN => "sin",
            OpType::COS => "cos",
            OpType::INPUT => "input",
            OpType::OUTPUT => "output",
            OpType::MULTIHEAD_ATTENTION => "multihead_attention",
            OpType::GETITEM => "getitem",
            OpType::GETATTR => "getattr",
            OpType::EXPAND => "expand",
            OpType::LAYER_NORM => "layer_norm",
            OpType::FLOOR_DIVIDE => "floor_divide",
            OpType::IDENTITY => "identity",
            OpType::GELU => "gelu",
            OpType::PERMUTE => "permute",
            OpType::SCALAR_MULTIPLY => "smultiply",
            OpType::SCALAR_FLOORDIV => "sfloordiv",
            OpType::SCALAR_ADD => "sadd",
            OpType::SCALAR_SUB => "ssub",
            OpType::SCALAR_TRUEDIV => "struediv",
            OpType::PARAMETER => "parameter",
            OpType::FLOAT => "float",
            OpType::CONTIGUOUS => "contigeous",
            OpType::TO => "to",
            OpType::UNSQUEEZE => "unsqueeze",
            OpType::TYPE_AS => "type_as",
            OpType::VIEW => "view",
            OpType::ATTRIBUTE => "attribute",
            OpType::EQ => "eq",
            OpType::SLICE => "slice",
            OpType::CALL => "call",
            OpType::ASSERT => "assert",
            OpType::CHUNK => "chunk",
            OpType::HARDSWISH => "hardswish",
            OpType::HARDSIGMOID => "hardsigmoid",
            OpType::SILU => "silu",
            OpType::MASKEDFILL => "masked_fill",
            OpType::REPEAT => "repeat",
            OpType::UNIFORM_LIKE => "uniform_like",
            OpType::LESS => "less",
            OpType::CAST => "cast",
            OpType::TENSOR => "tensor",
            OpType::MATMUL => "matmul",
            OpType::CLIP => "clip",
            OpType::ERF => "erf",
            OpType::BOOL => "bool",
            OpType::INVERT => "invert",
            OpType::AND => "And",
            OpType::DETACH => "detach",
            OpType::CUMSUM => "cumsum",
            OpType::ARANGE => "arange",
            // _ => panic!("Not supported operator!"),
        }
    }

    #[staticmethod]
    pub fn as_enum(id: &str) -> OpType {
        match id {
            "conv2d" => OpType::CONV2D,
            "embedding" => OpType::EMBEDDING,
            "pool2d" => OpType::POOL2D,
            "linear" => OpType::LINEAR,
            "softmax" => OpType::SOFTMAX,
            "concat" => OpType::CONCAT,
            "flat" => OpType::FLAT,
            "mseloss" => OpType::MSELOSS,
            "batchnorm" => OpType::BATCH_NORM,
            "relu" => OpType::RELU,
            "sigmoid" => OpType::SIGMOID,
            "tanh" => OpType::TANH,
            "elu" => OpType::ELU,
            "dropout" => OpType::DROPOUT,
            "batch_matmul" => OpType::BATCH_MATMUL,
            "split" => OpType::SPLIT,
            "reshape" => OpType::RESHAPE,
            "transpose" => OpType::TRANSPOSE,
            "reverse" => OpType::REVERSE,
            "exp" => OpType::EXP,
            "add" => OpType::ADD,
            "subtract" => OpType::SUBTRACT,
            "multiply" => OpType::MULTIPLY,
            "divide" => OpType::DIVIDE,
            "pow" => OpType::POW,
            "mean" => OpType::MEAN,
            "sqrt" => OpType::SQRT,
            "rsqrt" => OpType::RSQRT,
            "reciprocal" => OpType::RECIPROCAL,
            "neg" => OpType::NEG,
            "sin" => OpType::SIN,
            "cos" => OpType::COS,
            "input" => OpType::INPUT,
            "output" => OpType::OUTPUT,
            "multihead_attention" => OpType::MULTIHEAD_ATTENTION,
            "getitem" => OpType::GETITEM,
            "getattr" => OpType::GETATTR,
            "expand" => OpType::EXPAND,
            "layer_norm" => OpType::LAYER_NORM,
            "floor_divide" => OpType::FLOOR_DIVIDE,
            "identity" => OpType::IDENTITY,
            "gelu" => OpType::GELU,
            "permute" => OpType::PERMUTE,
            "smultiply" => OpType::SCALAR_MULTIPLY,
            "sfloordiv" => OpType::SCALAR_FLOORDIV,
            "sadd" => OpType::SCALAR_ADD,
            "ssub" => OpType::SCALAR_SUB,
            "struediv" => OpType::SCALAR_TRUEDIV,
            "parameter" => OpType::PARAMETER,
            "float" => OpType::FLOAT,
            "contigeous" => OpType::CONTIGUOUS,
            "to" => OpType::TO,
            "unsqueeze" => OpType::UNSQUEEZE,
            "type_as" => OpType::TYPE_AS,
            "view" => OpType::VIEW,
            "attribute" => OpType::ATTRIBUTE,
            "eq" => OpType::EQ,
            "call" => OpType::CALL,
            "assert" => OpType::ASSERT,
            "chunk" => OpType::CHUNK,
            "hardswish" => OpType::HARDSWISH,
            "hardsigmoid" => OpType::HARDSIGMOID,
            "silu" => OpType::SILU,
            "masked_fill" => OpType::MASKEDFILL,
            "repeat" => OpType::REPEAT,
            "uniform_like" => OpType::UNIFORM_LIKE,
            "less" => OpType::LESS,
            "cast" => OpType::CAST,
            "tensor" => OpType::TENSOR,
            "matmul" => OpType::MATMUL,
            "clip" => OpType::CLIP,
            "erf" => OpType::ERF,
            "bool" => OpType::BOOL,
            "invert" => OpType::INVERT,
            "And" => OpType::AND,
            "detach" => OpType::DETACH,
            "cumsum" => OpType::CUMSUM,
            "arange" => OpType::ARANGE,
            _ => {
                panic!("Not supported type!");
            }
        }
    }
}

// impl IntoPy<PyObject> for OpType {
//     fn into_py(self, py: Python<'_>) -> PyObject {
//         // delegates to i32's IntoPy implementation.
//         self.into_py(py)
//     }
// }
#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum ActiMode {
    AC_MODE_NONE = 10,
    AC_MODE_RELU,
    AC_MODE_SIGMOID,
    AC_MODE_TANH,
    AC_MODE_GELU,
}

#[pymethods]
impl ActiMode {
    pub fn as_int(&self) -> i32 {
        match self {
            ActiMode::AC_MODE_NONE => 10,
            ActiMode::AC_MODE_RELU => 11,
            ActiMode::AC_MODE_SIGMOID => 12,
            ActiMode::AC_MODE_TANH => 13,
            ActiMode::AC_MODE_GELU => 14,
        }
    }

    #[staticmethod]
    pub fn as_enum(id: i32) -> ActiMode {
        match id {
            10 => ActiMode::AC_MODE_NONE,
            11 => ActiMode::AC_MODE_RELU,
            12 => ActiMode::AC_MODE_SIGMOID,
            13 => ActiMode::AC_MODE_TANH,
            14 => ActiMode::AC_MODE_GELU,
            _ => {
                panic!("Not supported type!");
            }
        }
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum AggrMode {
    AGGR_MODE_NONE = 20,
    AGGR_MODE_SUM,
    AGGR_MODE_AVG,
}

#[pymethods]
impl AggrMode {
    pub fn as_int(&self) -> i32 {
        match self {
            AggrMode::AGGR_MODE_NONE => 20,
            AggrMode::AGGR_MODE_SUM => 21,
            AggrMode::AGGR_MODE_AVG => 22,
        }
    }

    #[staticmethod]
    pub fn as_enum(id: i32) -> AggrMode {
        match id {
            20 => AggrMode::AGGR_MODE_NONE,
            21 => AggrMode::AGGR_MODE_SUM,
            22 => AggrMode::AGGR_MODE_AVG,
            _ => {
                panic!("Not supported type!");
            }
        }
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum PoolType {
    POOL_MAX = 30,
    POOL_AVG,
    POOL_ADAPTIVE,
    POOL_ADAPTIVE_MAX,
    POOL_ADAPTIVE_AVG,
}

#[pymethods]
impl PoolType {
    pub fn as_int(&self) -> i32 {
        match self {
            PoolType::POOL_MAX => 30,
            PoolType::POOL_AVG => 31,
            PoolType::POOL_ADAPTIVE => 32,
            PoolType::POOL_ADAPTIVE_MAX => 33,
            PoolType::POOL_ADAPTIVE_AVG => 34,
        }
    }

    #[staticmethod]
    pub fn as_enum(id: i32) -> PoolType {
        match id {
            30 => PoolType::POOL_MAX,
            31 => PoolType::POOL_AVG,
            32 => PoolType::POOL_ADAPTIVE,
            33 => PoolType::POOL_ADAPTIVE_MAX,
            34 => PoolType::POOL_ADAPTIVE_AVG,
            _ => {
                panic!("Not supported type!");
            }
        }
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
#[repr(u32)]
pub enum ParamSyncType {
    NONE = 80,
    PS,
    NCCL,
}

// #[pyclass]
// #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
// #[repr(u32)]
// pub enum InitializerType {
//     ZeroInit,
//     NormInit,
//     UniformInit,
//     GlorotUniformInit,
// }

// #[pyclass]
// #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
// #[repr(u32)]
// pub enum OptimizerType {
//     SGD,
//     Adam,
// }

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum LossType {
    CATEGORICAL_CROSSENTROPY = 50,
    SPARSE_CATEGORICAL_CROSSENTROPY,
    MEAN_SQUARED_ERROR_AVG_REDUCE,
    MEAN_SQUARED_ERROR_SUM_REDUCE,
}

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum MetricsType {
    ACCURACY = 1001,
    CATEGORICAL_CROSSENTROPY,
    SPARSE_CATEGORICAL_CROSSENTROPY,
    MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_ERROR,
    MEAN_ABSOLUTE_ERROR,
}

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TryFromPrimitive)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum WeightType {
    EXTERNAL = 9001,
    INTERNAL,
}
