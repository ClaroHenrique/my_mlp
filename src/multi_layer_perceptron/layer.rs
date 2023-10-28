pub mod activation_functions;

pub struct Layer<'a> {
    pub weights: Vec<Vec<f32>>,
    pub d1: usize,
    pub d2: usize,

    pub activation: &'a dyn Fn(&mut Vec<f32>),
    pub activation_d: &'a dyn Fn(&mut Vec<f32>),

    pub result_foward: Vec<f32>,
    pub result_backward: Vec<f32>,
}

impl Layer<'_> {
    pub fn new(d1: usize, d2: usize, activation_name: i32) -> Layer<'static> {
        let weights = vec![vec![0.0; d2]; d1];
        let activation = &activation_functions::relu;
        let activation_d = &activation_functions::relu_d;
        let result_foward = vec![0.0; d2];
        let result_backward = vec![0.0; d1];

        Layer {
            d1: d1,
            d2: d2,
            weights: weights,
            activation: activation,
            activation_d: activation_d,
            result_foward: result_foward,
            result_backward: result_backward,
        }
    }

    pub fn foward(&mut self, input: &Vec<f32>) {
        for j in 0..(self.d2) {
            self.result_foward[j] = 0.0;
        }
        for i in 0..(self.d1) {
            for j in 0..(self.d2) {
                self.result_foward[j] += input[i] * self.weights[i][j];
            }
        }

        (self.activation)(&mut self.result_foward);
    }

    pub fn backward(&mut self, out_d: &mut Vec<f32>, learning_rate: f32) {
        (self.activation_d)(out_d);
    
        for i in 0..(self.d1) {
            self.result_backward[i] = 0.0;
        }
        for i in 0..(self.d1) {
            for j in 0..(self.d2) {
                self.result_backward[i] += out_d[j] * self.weights[i][j];
            }
        }

        // update weigths //
        for i in 0..(self.d1) {
            for j in 0..(self.d2) {
                self.weights[i][j] -= learning_rate * self.weights[i][j] * out_d[j];
            }
        }
    }
}



