pub mod layer;
use layer::Layer;

pub struct MultiLayerPerceptron<'a>{
    pub layers: Vec<Layer<'a>>,
    pub learning_rate: f32,
}

impl MultiLayerPerceptron<'_> {
    pub fn new(dims: Vec<usize>) -> MultiLayerPerceptron<'static>{
        let mut layers = Vec::new();

        for i in 1..dims.len(){
            let layer = Layer::new(dims[i-1], dims[i], 1);
            layers.push(layer);
        }

        MultiLayerPerceptron{
            layers: layers,
            learning_rate: 0.1,
        }
    }

    pub fn fit(&mut self, X: &Vec<Vec<f32>>, y: &Vec<f32>) {
        for i in 0..X.len() {
            self.fit_sample(&X[i], &y[i]);
        }
    }

    fn fit_sample(&mut self, x: &Vec<f32>, y: &f32) {
        let output = x.clone();
        let n = self.layers.len();

        self.layers[0].foward(&x);
        for i in 1..n {
            self.layers[i].foward(&self.layers[i-1].result_foward); //
        }

        let mut error = vec![(y - self.layers[n-1].result_foward[0])];
        self.layers[n-1].backward(&mut error, self.learning_rate);

        for i in (0..(n-1)).rev() {
            //
            //self.layers[i].backward(&mut self.layers[i+1].result_backward, self.learning_rate);
        }
    }
}

