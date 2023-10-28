
pub fn relu(x: &mut Vec<f32>) {
    for i in 0..x.len(){
        if x[i] < 0.0 {
            x[i] = 0.0;
        }  
    }
}

pub fn relu_d(x: &mut Vec<f32>) {
    for i in 0..x.len(){
        if x[i] >= 0.0 {
            x[i] = 1.0;
        } else {
            x[i] = 0.0;
        }  
    }
}


