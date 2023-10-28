mod multi_layer_perceptron;
use multi_layer_perceptron::MultiLayerPerceptron;


fn main() {
    println!("Hello, world!");

    let dims: Vec<usize> = vec![2, 3, 1];
    let learning_rate = 0.1;
    let mut mlp = MultiLayerPerceptron::new(dims, learning_rate);

    let X: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![3.0, 2.0, 1.0],
        vec![1.0, 2.0, 1.0],
    ];

    let y: Vec<f32> = vec![
        1.0,
        2.0,
        3.0,
    ];

    println!("mse: {}", mlp.mse(&X, &y));
    mlp.fit(&X, &y, 10);
    println!("mse: {}", mlp.mse(&X, &y));
    println!("{:?}", mlp.weights);
}
