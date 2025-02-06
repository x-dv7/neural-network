use crate::*;

// //входной слой
// (0.0,1,1,0),(1.0,1,1,1), (0.0,1,2,0),(1.0,1,2,2), (0.0,1,3,0),(1.0,1,3,3),//1,2,3
// //первый слой - 2 нейрона. 3 входа, 2 выхода
// (0.0,2,4,0),(-0.5,2,4,1),(-0.4,2,4,2),(-0.3,2,4,3),//4
// (0.0,2,5,0),(-0.2,2,5,1),(-0.1,2,5,2),(0.0,2,5,3),//5
// //второй слой - 1 нейрон. 2 входа, 1 выход
// (0.0,3,6,0),(-0.5,3,6,4),(0.5,3,6,5)//6
#[derive(Clone, Debug)]
pub struct FlexNetwork {
    /// -Список выходных нейронов (как ключи) со списками номеров
    ///  входных нейронов HashMap<usize, Vec<(usize,f32)>>.
    ///  для связей:  (usize,f32) = (№нейрона, вес)
    ///  для смещений:(usize,f32) = (0, bias)
    /// Смещение 1 слоя = 0.0, связь входного слоя одна и у неё (usize,f32) = (№нейрона, №входа),
    /// вес входной связи всегда = 1.0
    inp_links: HashMap<usize, Vec<(usize,f32)>>,
    /// -Список номеров нейронов по слоям Vec<Vec<usize>>
    neurons: Vec<Vec<usize>>,
    /// -Список функций активации каждого слоя HashMap<usize, candle_nn::Activation>
    activations: HashMap<usize, Activation>,
    /// -Список смещений нейронов по слоям Vec<Tensor>
    layers_biases: Vec<Tensor>,
    /// -Список весов нейронов по слоям Vec<Tensor>
    layers_weights: Vec<Tensor>,
}

impl FlexNetwork {
    /// Расчет в прямом направлении
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        //обнуляем входы, которые отключены и помещаем результат в новый вектор входов.
        //те входы, которые отключены имеют вес = 0.0, а не 1.0
        let mut l_inputs: Vec<f32> = Vec::new();
        for (j, num_inp) in self.neurons.first().unwrap().iter().enumerate() {//обходим все входы
            if let Some(inps) = self.inp_links.get(num_inp) {//поиск веса входа
                //вот он вход, смотрим его вес
                if let Some((_, w1)) = inps.iter().find(|(i, _)| *i == *num_inp) {
                    l_inputs.push(inputs[j] * *w1);
                }
            }
        }
        //переводим входной вектор в тензор
        let i_len = inputs.len();
        let xs = Tensor::new(l_inputs, &Device::Cpu).unwrap()
            .reshape((1, i_len)).unwrap();//(1,кол.вх.связей)
        //получаем модель нейронной сети из тензоров
        // let xs= xs.t().unwrap();
        let model = FlexNetwork::model(&self.layers_biases,
                                       &self.layers_weights,
                                       &self.activations).unwrap();
        //расчет выходов
        let last_layer_len = self.neurons.last().unwrap().len();
        let output_tensor = model.forward(&xs).unwrap();
        let output_tensor = output_tensor.reshape(last_layer_len).unwrap();
        let out: Vec<f32> = output_tensor.to_vec1().unwrap();
        out
    }
    /// Последовательность bias + веса входных связей и всё это послойно
    /// состав вых.структуры (bias or weight, layer_num, neuron_out, neuron_in)
    pub fn weights(&self) -> impl Iterator<Item = (f32, usize, usize, usize)> + '_ {
        let mut weights: Vec<(f32, usize, usize, usize)> = Vec::new();
        //обход послойно
        for (lnum, layer) in self.neurons.iter().enumerate() {
            //обход нейронов слоя
            for neuron_out in layer {
                if let Some(neurons_in) = self.inp_links.get(&(*neuron_out)) {
                    //обход входный связей и смещения нейрона
                    for (neuron_in, wt) in neurons_in {
                        weights.push((*wt, lnum+1, *neuron_out, *neuron_in));
                    }
                }
            }
        }
        weights.into_iter() // раскладываем всё в одну линию
    }
}

impl FlexNetwork {
    /// Конструктор
    pub fn new(
        inp_links: HashMap<usize, Vec<(usize,f32)>>,
        neurons: Vec<Vec<usize>>,
        activations: HashMap<usize, Activation>,
    ) -> Self {

        let (layers_biases, layers_weights)  =
            FlexNetwork::tensors(&neurons, &inp_links).unwrap();

        Self {
            inp_links,
            neurons,
            activations,
            layers_biases,
            layers_weights,
        }
    }
    /// Создание сети из весов (в них указана топология сети)
    pub fn from_weights(
        weights: impl IntoIterator<Item = (f32, usize, usize, usize)>
    ) -> Self {
        // let weights = weights.into_iter();
        //Список выходных нейронов (как ключи) со списками (номеров входных нейронов, весов) или
        //(0, смещение)
        let mut inp_links: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        //Список номеров нейронов по слоям
        let mut neurons: Vec<Vec<usize>> = Vec::new();
        //Список функций активации по слоям
        let mut activations: HashMap<usize, Activation> = HashMap::new();

        for (weight, layer_num, neuron_out, neuron_in) in weights {
            inp_links.entry(neuron_out).or_insert_with(Vec::new).push((neuron_in, weight));

            // Добавляем нейрон в соответствующий слой, если его еще нет
            if neurons.len() < layer_num {
                neurons.resize_with(layer_num, Vec::new);
            }
            let layer = &mut neurons[layer_num - 1];
            if !layer.contains(&neuron_out) {
                layer.push(neuron_out);
            }
            activations.entry(layer_num).or_insert(Activation::Relu); // Предполагаем, что по умолчанию Activation - Relu
        }
        Self::new(inp_links, neurons, activations)
    }
    /// Создание сети со случайными весами и указанной топологией
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopologyFlex]) -> Self {
        assert!(layers.len() > 1);
        let inp_links= FlexNetwork::input_links_rand(rng, layers);
        let neurons = FlexNetwork::neurons(layers);
        let activations = FlexNetwork::activations(layers);
        Self::new(inp_links, neurons, activations)
    }
    /// Входные связи для каждого нейрона
    pub fn input_links(layers: &[LayerTopologyFlex]) -> HashMap<usize, Vec<(usize,f32)>> {
        //список выходных нейронов (как ключи) со списками (номеров входных нейронов, весов) или
        //(0, смещение)
        layers
            .iter()//обход послойно
            .flat_map(|layer|
                        layer.connections.iter())//обход по связям слоя
            .fold(HashMap::new(),//новый список связей
                  |mut outs,
                   (weight, layer_num, neuron_out, neuron_in)|//одна связь
            {
                //добавляем новый вектор связей для нейрона (если его нет) и в вектор добавляем
                //номер вх.нейрона и вес связи с ним (или смещение для 0 вх.нейрона)
                outs.entry(*neuron_out).or_insert_with(Vec::new).push((*neuron_in,*weight));
                outs//список выходных нейронов (как ключи) со списками номеров входных нейронов
            })
    }
    /// Входные связи для каждого нейрона (заполнение весов и смещений случайными значениями)
    pub fn input_links_rand(rng: &mut dyn RngCore,
                            layers: &[LayerTopologyFlex]
    ) -> HashMap<usize, Vec<(usize,f32)>> {
        //список выходных нейронов (как ключи) со списками номеров входных нейронов
        layers
            .iter()//обход послойно
            .flat_map(|layer|
            layer.connections.iter())//обход по связям слоя
            .fold(HashMap::new(),//новый список связей
                  |mut outs,
                   (_, layer_num, neuron_out, neuron_in)|//одна связь
                  {
                      let wt: f32 =
                          if *layer_num == 1 {
                              if *neuron_in == 0 {
                                  0.0// Смещение 1 слоя
                              } else {
                                  1.0// Вес 1 слоя
                              }
                          } else {
                              rng.gen_range(-1.0..=1.0) // Вес или смещение
                          };
                      //добавляем новый вектор связей для нейрона (если его нет) и в вектор добавляем
                      //номер вх.нейрона и вес связи с ним (или смещение для 0 вх.нейрона)
                      outs.entry(*neuron_out).or_insert_with(Vec::new).push((*neuron_in, wt));
                      outs//список выходных нейронов (как ключи) со списками номеров входных нейронов
                  })
    }
    /// Список номеров нейронов по слоям (заполнение одноименного атрибута при создании сети)
    pub fn neurons(layers: &[LayerTopologyFlex]) -> Vec<Vec<usize>> {
        let mut neurons: Vec<Vec<usize>> = layers
            .iter()//обход послойно
            .map(|layer| {
                // список номеров нейронов в слое
                layer.connections
                    .iter()
                    .map(|(_, _, neuron_out, _)| *neuron_out)
                    .dedup()// Убираем дубликаты
                    .collect::<Vec<usize>>()//вектор нейронов в слое
            })
            .collect();
        //номера нейронов в слое д.б. по возрастанию
        neurons.iter_mut().for_each(|inner_vec| inner_vec.sort());
        neurons
    }
    /// Список функций активации каждого слоя
    pub fn activations(layers: &[LayerTopologyFlex]) -> HashMap<usize, Activation> {
        let mut a: HashMap<usize, Activation> = HashMap::new();
        for (l,layer) in layers.iter().enumerate() {
            a.insert(l+1,layer.activation);
        }
        a
    }
    /// Тензоры для расчета
    /// -Список смещений нейронов по слоям Vec<Tensor>
    /// -Список весов нейронов по слоям Vec<Tensor>
    pub fn tensors(neurons: &Vec<Vec<usize>>,
                   inp_links: &HashMap<usize, Vec<(usize,f32)>>,
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
                    l_count = inp.len() - 1;//кол. вх.связей тек. слоя
                    for (i, w1) in inp { //обход входных нейронов
                        if *i == 0 {
                            // для смещения (только одно на список):
                            b.push(*w1);
                        } else {
                            // для весов:
                            w.push(*w1);
                        }
                    }
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
                 layers_weights: &Vec<Tensor>,
                 act: &HashMap<usize, Activation>,
    ) -> Result<Sequential, candle_core::Error> {
        assert_eq!(layers_biases.len(), layers_weights.len());
        let mut model = seq();
        for (j, wt_mx) in layers_weights.iter().enumerate() {
            //слой линейный с активацией
            let model_layer = Linear::new(wt_mx.clone(), Some(layers_biases[j].clone())); //Some(bt_mx) или None
            model = model.add(model_layer);
            if let Some(func_act) = act.get(&j) {
                model = model.add(*func_act);
            } else {
                model = model.add(Relu);
            }
        }
        Ok(model)
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
    use candle_nn::{Linear}; //, Sequential

    #[test]
    fn model_check3() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        // первая сеть
        let mut connections1: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 1..=11 { // config.eye_cells = 9 + 2 (speed, rotation)
            connections1.push((0.0,1,i,0));//bias
            connections1.push((1.0,1,i,i));//weights
        };
        let mut connections2: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 12..=23 { // config.brain_neurons = 9 + 2 (speed, rotation)
            connections2.push((0.0,2,i,0));//bias
            for j in 1..=11 {
                connections2.push((0.0,2,i,j));//weights
            }
        };
        let mut connections3: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 24..=25 { // 2
            connections3.push((0.0,3,i,0));//bias
            for j in 12..=23 {
                connections3.push((0.0,3,i,j));//weights
            }
        };

        let topology = [
            LayerTopologyFlex {
                // neurons: 9 + 2, //config.eye_cells=9
                connections: connections1,
                activation: Activation::Relu,
            },
            LayerTopologyFlex {
                // neurons: 9 + 2, //config.brain_neurons=9
                connections: connections2,
                activation: Activation::Relu,
            },
            LayerTopologyFlex {
                // neurons: 2,
                connections: connections3,
                activation: Activation::Relu,
            },
        ];

        let net1 = FlexNetwork::random(&mut rng, &topology);
        let actual = net1.propagate(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]);

        Some(actual.len());

        //вторая сеть
        let mut connections1: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 1..=18 { // config.eye_cells = 9 * 2 вектора
            connections1.push((0.0,1,i,0));//bias
            connections1.push((1.0,1,i,i));//weights
        };
        let mut connections2: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 19..=37 { // config.brain_neurons = 9 * 2 вектора
            connections2.push((0.0,2,i,0));//bias
            for j in 1..=18 {
                connections2.push((0.0,2,i,j));//weights
            }
        };
        let mut connections3: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 38..=48 { // 2 + 9
            connections3.push((0.0,3,i,0));//bias
            for j in 19..=37 {
                connections3.push((0.0,3,i,j));//weights
            }
        };
        let topology = [
            LayerTopologyFlex {
                // neurons: 9 * 2,//config.eye_cells=9 * 2 вектора
                connections: connections1,
                activation: Activation::Relu,
            },
            LayerTopologyFlex {
                // neurons: 9 * 2,//config.brain_neurons=9 * 2 вектора
                connections: connections2,
                activation: Activation::Relu,
            },
            LayerTopologyFlex {
                // neurons: 2 + 9,
                connections: connections3,
                activation: Activation::Relu,
            },
        ];
        let net1 = FlexNetwork::random(&mut rng, &topology);
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
        let weights: Vec<(f32, usize, usize, usize)> =
            vec![//(вес,слой,нейрон,вх.связь)
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

        let network2 = FlexNetwork::from_weights(weights.clone());
        let actual = network2.propagate(vec![0.5f32, 0.7]);

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }

    #[test]
    fn weights() {
        let mut layers: Vec<LayerTopologyFlex> = Vec::new();
        let mut conn1: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 1..=2 { // config.eye_cells = 2
            conn1.push((0.0,1,i,0));//bias
            conn1.push((1.0,1,i,i));//weights
        };
        let layer = LayerTopologyFlex {
            // neurons: 2,
            connections: conn1,
            activation: Activation::Relu,
        };
        layers.push(layer);

        let mut conn2: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 3..=5 { // config.brain_neurons = 3
            conn2.push((0.0,2,i,0));//bias
            for j in 1..=2 {
                conn2.push((0.0,2,i,j));//weights
            }
        };
        let layer = LayerTopologyFlex {
            // neurons: 3,
            connections: conn2,
            activation: Activation::Relu,
        };
        layers.push(layer);

        let mut conn3: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 6..=6 { // 1
            conn3.push((0.0,3,i,0));//bias
            for j in 3..=5 {
                conn3.push((0.0,3,i,j));//weights
            }
        };
        let layer = LayerTopologyFlex {
            // neurons: 1,
            connections: conn3,
            activation: Activation::Relu,
        };
        layers.push(layer);

        let mut inp_links: HashMap<usize, Vec<(usize,f32)>> = HashMap::new();
        //вх.слой
        let mut layer_inp: Vec<(usize,f32)> = Vec::new();//1
        layer_inp.push((0, 0.1));
        layer_inp.push((1, 1.0));
        inp_links.insert(1, layer_inp);
        let mut layer_inp: Vec<(usize,f32)> = Vec::new();//2
        layer_inp.push((0, 0.2));
        layer_inp.push((2, 1.0));
        inp_links.insert(2, layer_inp);
        //1 слой
        let mut layer_inp: Vec<(usize,f32)> = Vec::new();//3
        layer_inp.push((0, 0.3));
        layer_inp.push((1, 0.4));
        layer_inp.push((2, 0.5));
        inp_links.insert(3, layer_inp);
        let mut layer_inp: Vec<(usize,f32)> = Vec::new();//4
        layer_inp.push((0, 0.6));
        layer_inp.push((1, 0.7));
        layer_inp.push((2, 0.8));
        inp_links.insert(4, layer_inp);
        let mut layer_inp: Vec<(usize,f32)> = Vec::new();//5
        layer_inp.push((0, 0.9));
        layer_inp.push((1, 1.0));
        layer_inp.push((2, 1.1));
        inp_links.insert(5, layer_inp);
        //2 слой
        let mut layer_inp: Vec<(usize,f32)> = Vec::new();//6
        layer_inp.push((0, 1.2));
        layer_inp.push((3, 1.3));
        layer_inp.push((4, 1.4));
        layer_inp.push((5, 1.5));
        inp_links.insert(6, layer_inp);

        let neurons = FlexNetwork::neurons(&layers);
        let activations = FlexNetwork::activations(&layers);

        let network = FlexNetwork::new(inp_links, neurons, activations);

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

        let actual: Vec<_> = FlexNetwork::from_weights(weights.clone())
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

        let network2 = FlexNetwork::from_weights(weights.clone());
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
}