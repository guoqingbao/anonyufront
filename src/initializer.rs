use pyo3::prelude::*;
use pyo3::pyclass;
use std::collections::HashMap;

#[pyclass]
pub struct Initializer {
    #[pyo3(get, set)]
    pub params: HashMap<String, String>,
}

#[pymethods]
impl Initializer {
    #[new]
    pub fn new(
        params: HashMap<String, String>,
    ) -> PyResult<PyClassInitializer<Self>> {
        println!("Initializer::new");
        let op = Initializer {
            params,
        };
        Ok(PyClassInitializer::from(op))
    }
}
