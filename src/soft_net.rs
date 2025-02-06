//NOT USED
use crate::*;

#[derive(Clone, Debug)]
pub struct SoftNetwork {
    /// Список смежности: HashMap<(usize, usize), f32>
    /// для смещения: HashMap<(neuron_out, 0), bias>
    /// Смещение 1 слоя = 0.0
    /// для весов:    HashMap<(neuron_out, neuron_in), вес>
    /// Вес входной связи всегда = 1.0
    /// neuron_in 1 слоя = номеру входа
    adjacency_list: HashMap<(usize, usize), f32>,
    /// Вектор функций активации каждого нейрона HashMap<usize, fn(f32) -> f32>
    activations: HashMap<usize, fn(f32) -> f32>,
    /// Топология сети
    layer_topologies: Vec<SoftLayerTopology>,
    /// Другой взгляд на топологию сети, необходим для propagate:
    /// рассчитывается при создании!
    /// -Список выходных нейронов (как ключи) со списками номеров
    ///  входных нейронов HashMap<usize, Vec<usize>>
    inp_links: HashMap<usize, Vec<usize>>,
    /// -Список номеров нейронов по слоям Vec<Vec<usize>>
    layers_out: Vec<Vec<usize>>,
    /// -Список смещений нейронов по слоям Vec<Tensor>
    layers_biases: Vec<Tensor>,
    /// -Список весов нейронов по слоям Vec<Tensor>
    layers_weights: Vec<Tensor>,
}


impl SoftNetwork {
    /// Конструктор
    pub fn new(
        adjacency_list: HashMap<(usize, usize), f32>, //HashMap<usize, Vec<(usize, f32)>>,
        activations: HashMap<usize, fn(f32) -> f32>,
        layer_topologies: Vec<SoftLayerTopology>,
    ) -> Self {
        let links = SoftNetwork::input_links(&layer_topologies);
        let neurons = SoftNetwork::neurons_out(&layer_topologies);
        let (layers_biases, layers_weights)  =
            SoftNetwork::tensors(&neurons, &links, &adjacency_list).unwrap();

        Self {
            adjacency_list,
            activations,
            layer_topologies, //: vec![]
            inp_links: links,
            layers_out: neurons,
            layers_biases,
            layers_weights,
        }
    }
    /// Создание сети со случайными весами и указанной топологией
    pub fn random(rng: &mut dyn RngCore, layers: &[SoftLayerTopology]) -> Self {
        assert!(layers.len() > 1);
        // Создание пустых: списка смещений, смежности, функций активации
        let mut adjacency_list: HashMap<(usize, usize), f32> = HashMap::new();
        let mut activations: HashMap<usize, fn(f32) -> f32> = HashMap::new();

        // Заполнение: списка смещений, смежности, функций активации
        let mut layer_num: usize = 1;
        for layer in layers {// обход послойно
            for (neuron_out, neuron_in) in &layer.connections {//обход по нейронам
                // Добавляем bias для текущего нейрона
                // делается 1 раз, поэтому добавляем заодно функцию активации
                if !adjacency_list.contains_key(&(*neuron_out, 0)) {
                    if layer_num == 1 {
                        adjacency_list.insert((*neuron_out, 0),0.0);// Смещение 1 слоя
                    } else {
                        let b1: f32 = rng.gen_range(-1.0..=1.0);
                        adjacency_list.insert((*neuron_out, 0), b1);// Смещение
                        activations.insert(neuron_out.clone(),
                                           |x: f32| x.max(0.0));//(ReLU)
                    }
                }
                // Добавляем связь от текущего нейрона к нейрону в списке connections
                if !adjacency_list.contains_key(&(neuron_out.clone(), neuron_in.clone())) {
                    if layer_num == 1 {
                        //у входного слоя прямое соединение от нейрона к входу
                        adjacency_list.insert((neuron_out.clone(), neuron_in.clone()),
                                              1.0);// Вес входной связи всегда = 1
                    }
                    else {
                        let w1: f32 = rng.gen_range(-1.0..=1.0);
                        adjacency_list.insert((neuron_out.clone(), neuron_in.clone()), w1);// Вес
                    }
                };
            }
            layer_num += 1;
        };
        Self::new(adjacency_list, activations, layers.to_vec()) //, total_neurons)
    }

    /// Входные связи для каждого нейрона (заполнение одноименного атрибута при создании сети)
    pub fn input_links(layers: &Vec<SoftLayerTopology>) -> HashMap<usize, Vec<usize>> {
        //список выходных нейронов (как ключи) со списками номеров входных нейронов
        layers
            .iter()//обход послойно
            .flat_map(|layer| layer.connections.iter())//обход по нейронам
            .fold(HashMap::new(), |mut outs, (neuron_out, neuron_in)|
            {
                outs.entry(*neuron_out).or_insert_with(Vec::new).push(*neuron_in);
                outs//список выходных нейронов (как ключи) со списками номеров входных нейронов
            })
    }
    /// Список номеров нейронов по слоям (заполнение одноименного атрибута при создании сети)
    pub fn neurons_out(layers: &Vec<SoftLayerTopology>) -> Vec<Vec<usize>> {
        layers
            .iter()//обход послойно
            .map(|layer| {
                // список номеров нейронов в слое
                layer.connections
                    .iter()
                    .map(|(neuron_out, _)| *neuron_out)
                    .dedup()// Убираем дубликаты
                    .collect::<Vec<usize>>()
            })
            .collect()
    }
    /// Тензоры для расчета
    /// -Список смещений нейронов по слоям Vec<Tensor>
    /// -Список весов нейронов по слоям Vec<Tensor>
    pub fn tensors(neurons: &Vec<Vec<usize>>,
                   inp_links: &HashMap<usize, Vec<usize>>,
                   adj_list: &HashMap<(usize, usize), f32>,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>), candle_core::Error> {
        let mut layers_biases: Vec<Tensor> = Vec::new();
        let mut layers_weights: Vec<Tensor> = Vec::new();
        let mut layer_num: usize = 1;
        let mut l_count: usize = 0;
        let mut n_count: usize = 0;
        for neurons_out in neurons { //обход послойно
            n_count = neurons_out.len();//кол.нейронов слоя
            if layer_num == 1 {
                layer_num += 1;
                continue;
            } //входной слой пропускаем
            let mut w: Vec<f32> = Vec::new();//веса входных связей одного слоя
            let mut b: Vec<f32> = Vec::new();//смещения одного слоя
            for (_, neuron_out) in neurons_out.iter().enumerate() {//обход нейронов слоя
                //веса
                if let Some(inp) = inp_links.get(neuron_out) {//список входных связей у нейрона
                    l_count = inp.len();//кол. вх.связей тек. слоя
                    for i in inp { //обход входных нейронов
                        // для весов:    HashMap<(neuron_out, neuron_in), вес>
                        if let Some(w1) = adj_list.get(&(*neuron_out, *i)) {
                            w.push(*w1);
                        }
                    }
                }
                //смещения
                // для смещения: HashMap<(neuron_out, 0), bias>
                if let Some(b1) = adj_list.get(&(*neuron_out, 0)) {
                    b.push(*b1);
                }
            }
            //смещения
            let bt_mx = Tensor::new(b, &Device::Cpu)?//смещения одного слоя
                .reshape((1, n_count))?;
            //веса
            let wt_mx = Tensor::new(w, &Device::Cpu)?//веса входных связей одного слоя
                .reshape((n_count, l_count))?;//(кол.нейр,кол.вх.связей)
            //докидываем к выходу
            layers_biases.push(bt_mx);
            layers_weights.push(wt_mx);
            layer_num += 1;
        }
        Ok((layers_biases, layers_weights))
    }
    /// Модель для расчета
    pub fn model(layers_biases: &Vec<Tensor>,
                 layers_weights: &Vec<Tensor>
    ) -> Result<Sequential, candle_core::Error> {
        assert_eq!(layers_biases.len(), layers_weights.len());
        let mut model = seq();
        for (j, wt_mx) in layers_weights.iter().enumerate() {
            //слой линейный с активацией
            let model_layer = Linear::new(wt_mx.clone(), Some(layers_biases[j].clone())); //Some(bt_mx) или None
            model = model.add(model_layer);
            model = model.add(Relu);
        }
        Ok(model)
    }

    /// Расчет в прямом направлении
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        // let mut out: Vec<f32> = vec![0.0f32; self.layer_topologies.last().unwrap().neurons];
        let i_len = inputs.len();
        let xs = Tensor::new(inputs, &Device::Cpu).unwrap()
            .reshape((1, i_len)).unwrap();//(1,кол.вх.связей)
        // let xs= xs.t().unwrap();
        let model = SoftNetwork::model(&self.layers_biases, &self.layers_weights).unwrap();
        let output_tensor = model.forward(&xs).unwrap()
            .reshape(self.layer_topologies.last().unwrap().neurons).unwrap();
        let out: Vec<f32> = output_tensor.to_vec1().unwrap();
        out
    }

    // pub fn from_weights(layers: &[LayerTopology], weights: impl IntoIterator<Item = f32>) -> Self {
    /// Создание сети из весов (в них указана топология сети)
    pub fn from_weights(
        weights: impl IntoIterator<Item = (f32, usize, usize, usize)>
    ) -> Self {
        // assert!(layers.len() > 1);
        let weights = weights.into_iter();

        let mut layers: Vec<SoftLayerTopology> = Vec::new();
        let mut adjacency_list: HashMap<(usize, usize), f32> = HashMap::new();
        let mut activations: HashMap<usize, fn(f32) -> f32> = HashMap::new();
        let mut layer_num: usize = 0;
        let mut neuron_out: usize = 0;
        for (w,l,o,i) in weights {
            if layer_num != l {//новый слой - добавляем его в вектор
                let layer_topology = SoftLayerTopology {
                    neurons: 0,
                    connections: Vec::new(),
                };
                layers.push(layer_topology);
                layer_num = layers.len();
            };
            if neuron_out != o {//поменялся нейрон
                // делается по 1 разу для каждого нейрона, поэтому добавляем функцию активации
                if layer_num != 1 {
                    activations.insert(o, |x: f32| x.max(0.0));//(ReLU)
                }
                if i == 0 {// первая запись при смене нейрона - смещение
                    if layer_num == 1 {
                        adjacency_list.insert((o, 0), 0.0);// Смещение 1 слоя
                    } else {
                        adjacency_list.insert((o, 0), w);// Смещение
                    }
                } else {// вдруг нет!
                    if layer_num == 1 {
                        adjacency_list.insert((o, i), 1.0);// Вес 1 слоя
                    } else {
                        adjacency_list.insert((o, i), w);// Вес
                    }
                    layers[layer_num - 1].connections.push((o, i));
                }
                neuron_out = o;
                layers[layer_num - 1].neurons += 1;//кол.нейронов в слое
            } else { //тот же самый нейрон
                if i != 0 {// последующие записи - веса
                    if layer_num == 1 {
                        adjacency_list.insert((o, i), 1.0);// Вес 1 слоя
                    } else {
                        adjacency_list.insert((o, i), w);// Вес
                    }
                    layers[layer_num - 1].connections.push((o, i));
                } else { // вдруг нет!
                    if layer_num == 1 {
                        adjacency_list.insert((o, 0), 0.0);// Смещение 1 слоя
                    } else {
                        adjacency_list.insert((o, 0), w);// Смещение
                    }
                }
            }
        }
        Self::new(adjacency_list, activations, layers)
    }

    /// Последовательность bias + веса входных связей и всё это послойно
    /// состав вых.структуры (bias or weight, layer_num, neuron_out, neuron_in)
    pub fn weights(&self) -> impl Iterator<Item = (f32, usize, usize, usize)> + '_ {
        let mut weights: Vec<(f32, usize, usize, usize)> = Vec::new();
        let mut layer_num: usize = 1;
        for layer in &self.layer_topologies {// послойный обход
            // if layer_num == 1 {continue};//первый слой не добавляем в веса
            let mut o: usize = 0;
            for (neuron_out, neuron_in) in &layer.connections {// обх.нейронов
                if o != *neuron_out {//нейрон поменялся
                    o = *neuron_out;
                    // сначала смещение нейрона
                    if let Some(bias) = self.adjacency_list.get(&(*neuron_out,0)) {
                        weights.push((*bias, layer_num, *neuron_out, 0));
                    }
                    // потом веса входных связей по очереди (здесь 1-й нейрон)
                    if let Some(wt) = self.adjacency_list.get(&(*neuron_out,*neuron_in)) {
                        weights.push((*wt, layer_num, *neuron_out, *neuron_in));
                    }
                }
                else {// потом веса входных связей по очереди
                    if let Some(wt) = self.adjacency_list.get(&(*neuron_out,*neuron_in)) {
                        weights.push((*wt, layer_num, *neuron_out, *neuron_in));
                    }
                }
            }
            layer_num += 1;
        }
        weights.into_iter()// раскладываем всё в одну линию
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use candle_core::{Device, Tensor};
    use candle_core::utils::has_accelerate; //, Result
    use candle_nn::{Linear};//, Sequential

    #[test]
    fn model_check3() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        // первая сеть
        let mut connections1: Vec<(usize, usize)> = Vec::new();
        for i in 1..=11 { // config.eye_cells = 9 + 2 (speed, rotation)
            connections1.push((i, i));
        };
        let mut connections2: Vec<(usize, usize)> = Vec::new();
        for i in 12..=23 { // config.brain_neurons = 9 + 2 (speed, rotation)
            for j in 1..=11 {
                connections2.push((i, j));
            }
        };
        let mut connections3: Vec<(usize, usize)> = Vec::new();
        for i in 24..=25 { // 2
            for j in 12..=23 {
                connections3.push((i, j));
            }
        };

        let topology = [
            SoftLayerTopology {
                neurons: 9 + 2, //config.eye_cells=9
                connections: connections1,
            },
            SoftLayerTopology {
                neurons: 9 + 2, //config.brain_neurons=9
                connections: connections2,
            },
            SoftLayerTopology {
                neurons: 2,
                connections: connections3,
            },
        ];

        let net1 = SoftNetwork::random(&mut rng, &topology);
        let actual = net1.propagate(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]);

        Some(actual.len());

        //вторая сеть
        let mut connections1: Vec<(usize, usize)> = Vec::new();
        for i in 1..=18 { // config.eye_cells = 9 * 2 вектора
            connections1.push((i, i));
        };
        let mut connections2: Vec<(usize, usize)> = Vec::new();
        for i in 19..=37 { // config.brain_neurons = 9 * 2 вектора
            for j in 1..=18 {
                connections2.push((i, j));
            }
        };
        let mut connections3: Vec<(usize, usize)> = Vec::new();
        for i in 38..=48 { // 2 + 9
            for j in 19..=37 {
                connections3.push((i, j));
            }
        };
        let topology = [
            SoftLayerTopology {
                neurons: 9 * 2,//config.eye_cells=9 * 2 вектора
                connections: connections1,
            },
            SoftLayerTopology {
                neurons: 9 * 2,//config.brain_neurons=9 * 2 вектора
                connections: connections2,
            },
            SoftLayerTopology {
                neurons: 2 + 9,
                connections: connections3,
            },
        ];
        let net1 = SoftNetwork::random(&mut rng, &topology);
        let actual = net1.propagate(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                         1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]);

        Some(actual.len());
    }

    #[test]
    fn model_check2() {
        //старая сеть - эталон
        let layers = (
            Layer::new(vec![
                Neuron::new(0.1, vec![0.2, 0.3]),
                Neuron::new(0.4, vec![0.5, 0.6]),
                Neuron::new(0.7, vec![0.8, 0.9]),
            ]),
            Layer::new(vec![
                Neuron::new(1.0, vec![1.1, 1.2, 1.3]),
                Neuron::new(1.4, vec![1.5, 1.6, 1.7]),
            ]),
        );
        let network1 = Network::new(vec![layers.0.clone(), layers.1.clone()]);
        let expected = network1.propagate(vec![0.5f32, 0.7]);

        //новая сеть (построим из весов, так легче)
        let weights: Vec<(f32, usize, usize, usize)> = vec![//(вес,слой,нейрон,вх.связь)
            //входной слой
            (0.0,1,1,0), (1.0,1,1,1),//1
            (0.0,1,2,0), (1.0,1,2,2),//2
            //первый слой - 3 нейрона. 2 входа, 3 выхода
            (0.1,2,3,0), (0.2,2,3,1),(0.3,2,3,2),//3
            (0.4,2,4,0), (0.5,2,4,1),(0.6,2,4,2),//4
            (0.7,2,5,0), (0.8,2,5,1),(0.9,2,5,2),//5
            //второй слой - 2 нейрона. 3 входа, 2 выхода
            (1.0,3,6,0), (1.1,3,6,3),(1.2,3,6,4),(1.3,3,6,5),//6
            (1.4,3,7,0), (1.5,3,7,3),(1.6,3,7,4),(1.7,3,7,5)//7
        ];

        let network2 = SoftNetwork::from_weights(weights.clone());
        let actual = network2.propagate(vec![0.5f32, 0.7]);

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }

    #[test]
    fn weights() {
        let mut layers: Vec<SoftLayerTopology> = Vec::new();
        let mut conn1: Vec<(usize, usize)> = Vec::new();
        for i in 1..=2 { // config.eye_cells = 2
            conn1.push((i, i));
        };
        let layer = SoftLayerTopology {
            neurons: 2,
            connections: conn1,
        };
        layers.push(layer);

        let mut conn2: Vec<(usize, usize)> = Vec::new();
        for i in 3..=5 { // config.brain_neurons = 3
            for j in 1..=2 {
                conn2.push((i, j));
            }
        };
        let layer = SoftLayerTopology {
            neurons: 3,
            connections: conn2,
        };
        layers.push(layer);

        let mut conn3: Vec<(usize, usize)> = Vec::new();
        for i in 6..=6 { // 1
            for j in 3..=5 {
                conn3.push((i, j));
            }
        };
        let layer = SoftLayerTopology {
            neurons: 1,
            connections: conn3,
        };
        layers.push(layer);

        let mut adjacency_list: HashMap<(usize, usize), f32> = HashMap::new();
        adjacency_list.insert((1,0), 0.1);//1
        adjacency_list.insert((1,1), 1.0);
        adjacency_list.insert((2,0), 0.2);//2
        adjacency_list.insert((2,2), 1.0);
        adjacency_list.insert((3,0), 0.3);//3
        adjacency_list.insert((3,1), 0.4);
        adjacency_list.insert((3,2), 0.5);
        adjacency_list.insert((4,0), 0.6);//4
        adjacency_list.insert((4,1), 0.7);
        adjacency_list.insert((4,2), 0.8);
        adjacency_list.insert((5,0), 0.9);//5
        adjacency_list.insert((5,1), 1.0);
        adjacency_list.insert((5,2), 1.1);
        adjacency_list.insert((6,0), 1.2);//6
        adjacency_list.insert((6,3), 1.3);
        adjacency_list.insert((6,4), 1.4);
        adjacency_list.insert((6,5), 1.5);
        let mut activations: HashMap<usize, fn(f32) -> f32> = HashMap::new();
        activations.insert(1, |x: f32| x.max(0.0));//(ReLU)
        activations.insert(2, |x: f32| x.max(0.0));//(ReLU)
        activations.insert(3, |x: f32| x.max(0.0));//(ReLU)
        activations.insert(4, |x: f32| x.max(0.0));//(ReLU)
        activations.insert(5, |x: f32| x.max(0.0));//(ReLU)
        activations.insert(6, |x: f32| x.max(0.0));//(ReLU)

        let network = SoftNetwork::new(adjacency_list, activations, layers);

        let actual: Vec<(f32, usize, usize, usize)> = network.weights().collect();
        let mut act_w: Vec<f32> = Vec::new();
        let mut act_l: Vec<usize> = Vec::new();
        let mut act_o: Vec<usize> = Vec::new();
        let mut act_i: Vec<usize> = Vec::new();
        for (w,l,o,i) in actual {
            act_w.push(w);
            act_l.push(l);
            act_o.push(o);
            act_i.push(i);
        }
        let expected:Vec<(f32, usize, usize, usize)> = vec![
                            (0.1,1,1,0), (1.0,1,1,1), (0.2,1,2,0),(1.0,1,2,2),//1,2
                            (0.3,2,3,0),(0.4,2,3,1),(0.5,2,3,2),//3
                            (0.6,2,4,0),(0.7,2,4,1),(0.8,2,4,2),//4
                            (0.9,2,5,0),(1.0,2,5,1),(1.1,2,5,2),//5
                            (1.2,3,6,0),(1.3,3,6,3),(1.4,3,6,4),(1.5,3,6,5)//6
                            ];
        let mut exp_w: Vec<f32> = Vec::new();
        let mut exp_l: Vec<usize> = Vec::new();
        let mut exp_o: Vec<usize> = Vec::new();
        let mut exp_i: Vec<usize> = Vec::new();
        for (w,l,o,i) in expected {
            exp_w.push(w);
            exp_l.push(l);
            exp_o.push(o);
            exp_i.push(i);
        }
        assert_relative_eq!(act_w.as_slice(), exp_w.as_slice());
        assert_eq!(act_l.as_slice(), exp_l.as_slice());
        assert_eq!(act_o.as_slice(), exp_o.as_slice());
        assert_eq!(act_i.as_slice(), exp_i.as_slice());
    }

    #[test]
    fn from_weights() {
        let weights:Vec<(f32, usize, usize, usize)> = vec![
            (0.0,1,1,0), (1.0,1,1,1), (0.0,1,2,0),(1.0,1,2,2),//1,2
            (0.3,2,3,0),(0.4,2,3,1),(0.5,2,3,2),//3
            (0.6,2,4,0),(0.7,2,4,1),(0.8,2,4,2),//4
            (0.9,3,5,0),(1.0,3,5,3),(1.1,3,5,4)//5
        ];

        let actual: Vec<_> = SoftNetwork::from_weights(weights.clone())
            .weights()
            .collect();
        let mut act_w: Vec<f32> = Vec::new();
        let mut act_l: Vec<usize> = Vec::new();
        let mut act_o: Vec<usize> = Vec::new();
        let mut act_i: Vec<usize> = Vec::new();
        for (w,l,o,i) in actual {
            act_w.push(w);
            act_l.push(l);
            act_o.push(o);
            act_i.push(i);
        }

        let mut exp_w: Vec<f32> = Vec::new();
        let mut exp_l: Vec<usize> = Vec::new();
        let mut exp_o: Vec<usize> = Vec::new();
        let mut exp_i: Vec<usize> = Vec::new();
        for (w,l,o,i) in weights {
            exp_w.push(w);
            exp_l.push(l);
            exp_o.push(o);
            exp_i.push(i);
        }

        assert_relative_eq!(act_w.as_slice(), exp_w.as_slice());
        assert_eq!(act_l.as_slice(), exp_l.as_slice());
        assert_eq!(act_o.as_slice(), exp_o.as_slice());
        assert_eq!(act_i.as_slice(), exp_i.as_slice());
    }

    #[test]
    fn propagate() {
        //сравним результаты старой сети и новой
        //старая сеть - эталон
        let layers = (
            Layer::new(vec![
                Neuron::new(0.0, vec![-0.5, -0.4, -0.3]),
                Neuron::new(0.0, vec![-0.2, -0.1, 0.0]),
            ]),
            Layer::new(vec![Neuron::new(0.0, vec![-0.5, 0.5])]),
        );
        let network1 = Network::new(vec![layers.0.clone(), layers.1.clone()]);
        let expected = network1.propagate(vec![0.5, 0.6, 0.7]);
        // let wts: Vec<f32> = network1.weights().collect();

        //новая сеть (построим из весов, так легче)
        let weights:Vec<(f32, usize, usize, usize)> = vec![
            //входной слой
            (0.0,1,1,0),(1.0,1,1,1), (0.0,1,2,0),(1.0,1,2,2), (0.0,1,3,0),(1.0,1,3,3),//1,2,3
            //первый слой - 2 нейрона. 3 входа, 2 выхода
            (0.0,2,4,0),(-0.5,2,4,1),(-0.4,2,4,2),(-0.3,2,4,3),//4
            (0.0,2,5,0),(-0.2,2,5,1),(-0.1,2,5,2),(0.0,2,5,3),//5
            //второй слой - 1 нейрон. 2 входа, 1 выход
            (0.0,3,6,0),(-0.5,3,6,4),(0.5,3,6,5)//6
        ];

        let network2 = SoftNetwork::from_weights(weights.clone());
        let actual = network2.propagate(vec![0.5, 0.6, 0.7]);

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }

    #[test]
    fn model_check() {
        //сравним результаты старой сети и новой
        //старая сеть - эталон
        let layers = (
            Layer::new(vec![
                Neuron::new(0.0, vec![-0.5, -0.4, -0.3]),
                Neuron::new(0.0, vec![-0.2, -0.1, 0.0]),
            ]),
            Layer::new(vec![Neuron::new(0.0, vec![-0.5, 0.5])]),
        );
        let network1 = Network::new(vec![layers.0.clone(), layers.1.clone()]);
        let expected = network1.propagate(vec![0.5, 0.6, 0.7]);

        // //новая сеть
        // struct Model {
        //     layers: Vec<Tensor>,
        // }
        //
        // impl Model {
        //     fn forward(&self, image: &Tensor) -> Result<Tensor> {
        //         let x = image.matmul(&self.layers[0])?; //=[1, 2]
        //         let x = x.reshape(2)?;
        //         let x = x.relu()?;
        //         let x = x.reshape((1, 2))?;
        //         x.matmul(&self.layers[1])
        //     }
        // }
        //
        // let mut layers_new: Vec<Tensor> = Vec::new();
        // //смещения
        // // let b = vec![]
        // // let bt_mx = Tensor::new(b, &Device::Cpu).unwrap();//смещения одного слоя
        // // let bt_mx = bt_mx.reshape((1,n_count)).unwrap();
        // //веса
        // let w1 = vec![-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0];
        // let wt_mx1 = Tensor::new(w1, &Device::Cpu).unwrap();//веса входных связей одного слоя
        // let wt_mx1 = wt_mx1.reshape((3, 2)).unwrap();//изменения размерности тензора (стр.,столбц.)
        // layers_new.push(wt_mx1);
        //
        // let w2 = vec![-0.5f32, 0.5];
        // let wt_mx2 = Tensor::new(w2, &Device::Cpu).unwrap();//веса входных связей одного слоя
        // let wt_mx2 = wt_mx2.reshape((2, 1)).unwrap();//изменения размерности тензора
        // layers_new.push(wt_mx2);
        //
        // let model = Model { layers: layers_new };
        //
        // let dummy_image = Tensor::new(vec![0.5f32, 0.6, 0.7], &Device::Cpu).unwrap();
        // let dummy_image = dummy_image.reshape((1, 3)).unwrap();//изменения размерности тензора
        //
        // let actual = model.forward(&dummy_image).unwrap();
        // let actual = actual.reshape(1).unwrap();
        // let actual = actual.to_vec1::<f32>().unwrap();

        // новая сеть
        // Создаем слои
        let w1 = vec![-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0];
        let wt_mx1 = Tensor::new(w1, &Device::Cpu).unwrap()
            .reshape((2, 3)).unwrap();//веса входных связей одного слоя (кол.нейр,кол.вх.связей)
        let layer1 = Linear::new(wt_mx1, None);

        let w2 = vec![-0.5f32, 0.5];
        let wt_mx2 = Tensor::new(w2, &Device::Cpu).unwrap()
            .reshape((1, 2)).unwrap();//веса входных связей одного слоя (кол.нейр,кол.вх.связей)
        let layer2 = Linear::new(wt_mx2, None);

        // Создаем модель
        let model = seq();
        let model = model.add(layer1);
        let model = model.add(Relu);
        let model = model.add(layer2);
        // Входные данные
        let dummy_image = Tensor::new(vec![0.5f32, 0.6, 0.7], &Device::Cpu).unwrap()
            .reshape((1, 3)).unwrap();//(1,кол.вх.связей)
        // Вычисление выхода
        let actual = model.forward(&dummy_image).unwrap();
        let actual = actual.reshape(1).unwrap();
        let actual = actual.to_vec1::<f32>().unwrap();

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }

    // #[test]
    // fn random() {
    //     let mut rng = ChaCha8Rng::from_seed(Default::default());
    //
    //     let network = Network::random(
    //         &mut rng,
    //         &[
    //             LayerTopology { neurons: 3 },
    //             LayerTopology { neurons: 2 },
    //             LayerTopology { neurons: 1 },
    //         ],
    //     );
    //
    //     assert_eq!(network.layers.len(), 2);
    //     assert_eq!(network.layers[0].neurons.len(), 2);
    //
    //     assert_relative_eq!(network.layers[0].neurons[0].bias, -0.6255188);
    //
    //     assert_relative_eq!(
    //         network.layers[0].neurons[0].weights.as_slice(),
    //         &[0.67383957, 0.8181262, 0.26284897].as_slice()
    //     );
    //
    //     assert_relative_eq!(network.layers[0].neurons[1].bias, 0.5238807);
    //
    //     assert_relative_eq!(
    //         network.layers[0].neurons[1].weights.as_slice(),
    //         &[-0.5351684, 0.069369555, -0.7648182].as_slice()
    //     );
    //
    //     assert_eq!(network.layers[1].neurons.len(), 1);
    //
    //     assert_relative_eq!(
    //         network.layers[1].neurons[0].weights.as_slice(),
    //         &[-0.48879623, -0.19277143].as_slice()
    //     );
    // }
    //
}

// СТАРЫЙ Расчет в прямом направлении
// pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
//     // // let mut outs1 = Tensor::new(vec![0.0; self.layer_topologies[0].neurons], &Device::Cpu); // Выходные значения первого слоя
//     let mut outs: HashMap<usize, f32> = HashMap::new(); // выходные значения
//     let mut outputs: Vec<f32> = Vec::new(); // выходной вектор
//     let mut layer_num: usize = 1;
//     let last_layer_num: usize = self.layer_topologies.len();
//     let mut neurons_out: Vec<usize> = Vec::new(); // список номеров нейронов в слое
// let mut o: usize = 0;
// //составим список нейронов слоя
// neurons_out.clear();
// for (neuron_out, _) in &layer.connections {
//     if *neuron_out != o {
//         neurons_out.push(*neuron_out);//список номеров нейронов
//         o = *neuron_out;
//     };
// };
//
    // for layer in &self.layer_topologies { //обход послойно
    //     //1 обход по нейронам для взвешенных сумм
    //     // let mut values = Tensor::new(vec![0.0; layer.neurons], &Device::Cpu); // Значения нейронов в слое
    //     let mut o: usize = 0;
    //     neurons_out.clear();
    //     for (neuron_out, neuron_in) in &layer.connections {
    //         if *neuron_out != o {
    //             neurons_out.push(*neuron_out);//список номеров нейронов
    //             o = *neuron_out;
    //         };
    //         let mut value: f32; //значение нейрона
    //         if layer_num == 1 { //для 1 слоя берем значения из входов
    //             value = inputs[*neuron_in - 1].clone();
    //             outs.insert(*neuron_out, value); //добавляем значение нейрона
    //         } else { //другие слои
    //             // value = if let Some(v) = outs.get(neuron_in) { *v } else { 0.0 };
    //             value = outs.get(neuron_in).cloned().unwrap_or(0.0);
    //             if let Some(wt) = self.adjacency_list.get(&(*neuron_out, *neuron_in)) {
    //                 value = value * wt;
    //             };
    //             if let Some(old) = outs.insert(*neuron_out, value) {
    //                 value = old + value;
    //             };
    //             outs.insert(*neuron_out, value); //добавляем значение нейрона
    //         }
    //     }
    //     //2 обход по нейронам для смещений и функции активации
    //     for neuron_out in &neurons_out {
    //         let mut value: f32; //значение нейрона
    //         value = if let Some(v) = outs.get(neuron_out) { *v } else { 0.0 };
    //         if let Some(bias) = self.adjacency_list.get(&(*neuron_out, 0)) {
    //             value = outs[neuron_out] + bias;
    //         }
    //         if let Some(act) = self.activations.get(neuron_out) {
    //             value = act(value); // Применение функции активации
    //         }
    //         if layer_num == last_layer_num {
    //             outputs.push(value); //Формируем выход
    //         } else {
    //             outs.insert(*neuron_out, value); //возвращем значение нейрона
    //         }
    //     }
    //     layer_num += 1;
    // }
//     outputs
// }

// /// Расчет в прямом направлении
// pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32>
// {
//     // // let mut outs1 = Tensor::new(vec![0.0f32; *max_inp_links + 1], &Device::Cpu); // Выходные значения первого слоя
//     let mut outputs: Vec<f32> = Vec::new(); // выходной вектор
//     let mut layer_num: usize = 1;
//     let last_layer_num: usize = self.layer_topologies.len();
//     //
//     let max_inp_links = self.inp_links.keys().max().unwrap();
//     let mut outs: Vec<f32> = vec![0.0f32; *max_inp_links + 1];
//     // let mut outs = Tensor::new(vec![0.0f32; *max_inp_links + 1], &Device::Cpu).unwrap();
//
//     for neurons_out in &self.layers_out { //обход послойно
//         // neurons_out - список номеров нейронов в слое
//         for (j, neuron_out) in neurons_out.iter().enumerate() {//обход нейронов слоя
//         // for neuron_out in neurons_out { //обход нейронов слоя
//             let mut value: f32 =  //значение нейрона
//                 // Взвешенная сумма входов
//                 if let Some(inp) = self.inp_links.get(neuron_out) {
//                     let mut s: f32 = 0.0f32;//взвешенная сумма
//                     for i in inp {//обход входных нейронов
//                         let v: f32 =
//                             if layer_num == 1 { //для 1 слоя берем значения из входов
//                                 inputs[*i - 1].clone()
//                             } else { //другие слои
//                                 outs[*i].clone()
//                             };
//                         // для весов:    HashMap<(neuron_out, neuron_in), вес>
//                         if let Some(wt) = self.adjacency_list.get(&(*neuron_out, *i)) {
//                             s = s + (v * *wt);
//                         } else {
//                             s = s + v;//вес не нашли (на входе), суммируем без него
//                         };
//                     }
//                     s
//                 } else { 0.0f32 };
//             // let mut value = Tensor::try_from(0.0f32).unwrap(); //new(vec![0.0f32], &Device::Cpu);//значение нейрона
//             // // Взвешенная сумма входов
//             // if let Some(inp) = self.inp_links.get(neuron_out) {
//             //     for i in inp { // обход входных нейронов
//             //         let v =
//             //             if layer_num == 1 {//для 1 слоя берем значения из входов
//             //                 Tensor::try_from(inputs[*i - 1]).unwrap()
//             //                 // Tensor::new(vec![inputs[*i - 1]], &Device::Cpu).unwrap()
//             //             } else {//другие слои - выходы других нейронов
//             //                 Tensor::try_from(outs[*i].clone()).unwrap()
//             //                 // Tensor::new(vec![outs[*i].clone()], &Device::Cpu).unwrap()
//             //             };
//             //         // для весов:    HashMap<(neuron_out, neuron_in), вес>
//             //         if let Some(wt) = self.adjacency_list.get(&(*neuron_out, *i)) {
//             //             // let wt_t = Tensor::new(vec![*wt], &Device::Cpu).unwrap();
//             //             // let wt_t = Tensor::try_from(*wt).unwrap();
//             //             value = value.add(&v.mul(&Tensor::try_from(*wt).unwrap()).unwrap()).unwrap();
//             //         } else {
//             //             value = value.add(&v).unwrap(); // вес не нашли (на входе), суммируем без него
//             //         };
//             //     }
//             // }
//             // для смещения: HashMap<(neuron_out, 0), bias>
//             if let Some(bias) = self.adjacency_list.get(&(*neuron_out, 0)) {
//                 value = value + *bias;
//             }
//             // if let Some(bias) = self.adjacency_list.get(&(*neuron_out, 0)) {
//             //     value = value.add(&Tensor::try_from(*bias).unwrap()).unwrap();
//             // }
//             // Применение функции активации
//             if let Some(act) = self.activations.get(neuron_out) {
//                 value = act(value);
//                 // value = value.relu().unwrap(); // Применяем Relu
//             }
//             // возвращаем значение нейрона
//             // outs[*neuron_out] = value.to_scalar().unwrap(); //возвращем значение нейрона
//             if layer_num == last_layer_num {
//                 // outputs.push(value.to_scalar().unwrap()); //Формируем выход
//                 outputs.push(value); //Формируем выход
//             }
//         }
//         layer_num += 1;
//     }
//     outputs
// }

// pub fn model(neurons: &Vec<Vec<usize>>,
//              inp_links: &HashMap<usize, Vec<usize>>,
//              adj_list: &HashMap<(usize, usize), f32>,
// ) -> Result<Sequential, candle_core::Error> {
//     let mut model = seq();
//     let mut layer_num: usize = 1;
//     let mut l_count: usize = 0;
//     let mut n_count: usize = 0;
//     for neurons_out in neurons { //обход послойно
//         n_count = neurons_out.len();//кол.нейронов слоя
//         if layer_num == 1 {
//             layer_num += 1;
//             continue;
//         } //входной слой пропускаем
//         let mut w: Vec<f32> = Vec::new();//веса входных связей одного слоя
//         let mut b: Vec<f32> = Vec::new();//смещения одного слоя
//         for (_, neuron_out) in neurons_out.iter().enumerate() {//обход нейронов слоя
//             //веса
//             if let Some(inp) = inp_links.get(neuron_out) {//список входных связей у нейрона
//                 l_count = inp.len();//кол. вх.связей тек. слоя
//                 for i in inp { //обход входных нейронов
//                     // для весов:    HashMap<(neuron_out, neuron_in), вес>
//                     if let Some(w1) = adj_list.get(&(*neuron_out, *i)) {
//                         w.push(*w1);
//                     }
//                 }
//             }
//             //смещения
//             // для смещения: HashMap<(neuron_out, 0), bias>
//             if let Some(b1) = adj_list.get(&(*neuron_out, 0)) {
//                 b.push(*b1);
//             }
//         }
//         //смещения
//         let bt_mx = Tensor::new(b, &Device::Cpu)?//смещения одного слоя
//             .reshape((1, n_count))?;
//         //веса
//         let wt_mx = Tensor::new(w, &Device::Cpu)?//веса входных связей одного слоя
//             .reshape((n_count, l_count))?;//(кол.нейр,кол.вх.связей)
//         //слой линейный с активацией
//         let model_layer = Linear::new(wt_mx, Some(bt_mx)); //Some(bt_mx) или None
//         model = model.add(model_layer);
//         model = model.add(Relu);
//         layer_num += 1;
//     }
//     Ok(model)
// }