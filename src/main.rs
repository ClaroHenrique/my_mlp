mod multi_layer_perceptron;
use multi_layer_perceptron::MultiLayerPerceptron;


fn main() {
    println!("Hello, world!");

    let dims: Vec<usize> = vec![2, 3, 1];
    let mlp = MultiLayerPerceptron::new(dims);

    println!("{:?}", mlp.layers[0].weights);
}
