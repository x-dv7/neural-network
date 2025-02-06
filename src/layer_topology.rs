use crate::*;

#[derive(Clone, Copy, Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}

/// Топология слоя
#[derive(Clone, Debug)]
pub struct LayerTopologySoft { //NOT USED
    /// Количество нейронов в слое
    pub neurons: usize,
    /// Список номеров нейронов слоя
    /// +
    /// Список входных связей каждого нейрона (список смежности):
    /// (neuron_out, neuron_in)
    /// 1-й слой ни на что не ссылается, т.е. Vec c 1 записью.
    /// Там указано прямое соответствие: на номер входа.
    /// (neuron_out, input)
    pub connections: Vec<(usize, usize)>,
}

/// Топология слоя
#[derive(Clone, Debug)]
pub struct LayerTopologyFlex {
    // /// Количество нейронов в слое
    // pub neurons: usize,
    /// Состав структуры (bias or weight, layer_num, neuron_out, neuron_in),
    /// совпадает с Chromosome.genes
    pub connections: Vec<(f32, usize, usize, usize)>,
    /// Функция активации слоя
    pub activation: Activation,
}