pub mod activation_functions;
use rand::Rng;

pub struct MultiLayerPerceptron<'a>{
    pub weights: Vec<Vec<Vec<f32>>>,
    pub weights_d: Vec<Vec<Vec<f32>>>,

    pub activations: Vec<&'a dyn Fn(&mut Vec<f32>)>,
    pub activations_d: Vec<&'a dyn Fn(&mut Vec<f32>)>,

    pub learning_rate: f32,
}

impl MultiLayerPerceptron<'_> {
    pub fn new(dims: Vec<usize>, learning_rate: f32) -> MultiLayerPerceptron<'static>{
        let mut weights = Vec::new();
        let mut weights_d = Vec::new();

        // let mut activations = Vec::new();
        // let mut activations_d = Vec::new();

        for i in 1..dims.len(){
            let layer = vec![vec![0.0; dims[i]]; dims[i-1]];
            let layer_d = vec![vec![0.0; dims[i]]; dims[i-1]];

            weights.push(layer);
            weights_d.push(layer_d);

            // activations.push(&activation_functions::relu);
            // activations_d.push(&activation_functions::relu_d);
        }

        let mut rng = rand::thread_rng();
        for l in 0..weights.len(){
            for i in 0..weights[l].len(){
                for j in 0..weights[l][i].len() {
                    weights[l][i][j] = rng.gen_range(-1.0..=1.0);
                }
            }
        }

        MultiLayerPerceptron{
            weights: weights,
            weights_d: weights_d,
            activations: vec![&activation_functions::relu; dims.len()],
            activations_d: vec![&activation_functions::relu_d; dims.len()],
            learning_rate: learning_rate,
        }
    }

    pub fn foward(&self, input: &Vec<f32>) -> Vec<f32>{
        let mut layer_input = input.clone();
        let mut layer_output = Vec::new();

        for l in 0..self.weights.len() {
            let d1 = self.weights[l].len();
            let d2 = self.weights[l][1].len();

            layer_output = vec![0.0; d2];
            
            for i in 0..d1 {
                for j in 0..d2 {
                    layer_output[j] += layer_input[i] * self.weights[l][i][j];
                }
            }
            (self.activations[l])(&mut layer_output);
            layer_input = layer_output.clone();
        }
        layer_output
    }

    pub fn backward(&mut self, error: &Vec<f32>) -> Vec<f32>{
        let mut layer_input = error.clone();
        let mut layer_output = Vec::new();

        for l in (0..self.weights.len()).rev() {
            let d1 = self.weights[l].len();
            let d2 = self.weights[l][1].len();

            layer_output = vec![0.0; d1];
            
            for i in 0..d1 {
                for j in 0..d2 {
                    let d = layer_input[j] * self.weights[l][i][j];
                    self.weights_d[l][i][j] = d;
                    layer_output[i] += d;
                }
            }
            (self.activations_d[l])(&mut layer_output);
            layer_input = layer_output.clone();
        }
        layer_output
    }

    pub fn fit(&mut self, X: &Vec<Vec<f32>>, y: &Vec<f32>, epochs: usize) {
        for _ in 0..epochs{
            for i in 0..X.len() {
                self.fit_sample(&X[i], &y[i]);
            }
        }
    }

    fn fit_sample(&mut self, x: &Vec<f32>, y: &f32) {
        let mut input = x.clone();

        // foward
        let mut foward_result = self.foward(&input);

        // calculate error
        let error = vec![foward_result[0] - y];

        // backward
        self.backward(&error);

        // update weigths
        for l in 0..self.weights.len(){
            for i in 0..self.weights[l].len(){
                for j in 0..self.weights[l][i].len(){
                    self.weights[l][i][j] -= self.learning_rate * self.weights_d[l][i][j];
                }
            }
        }
    }

    pub fn predict(&self, X: &Vec<Vec<f32>>) -> Vec<f32> {
        let mut result = vec![0.0; X.len()];
        for i in 0..X.len() {
            result[i] = self.foward(&X[i])[0];
        }

        result
    }

    pub fn mse(&self, X: &Vec<Vec<f32>>, y: &Vec<f32>) -> f32{
        let mut result = 0.0;
        let n = X.len();
        let ŷ = self.predict(X);

        for i in 0..n{
            result += (y[i] - ŷ[i]).powf(2.0);
        }
        result / (n as f32)
    }
}

