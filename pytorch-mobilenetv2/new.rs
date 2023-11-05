#[no_mangle]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[no_mangle]
pub fn solve_hard_problem() {
    let solution = add(40, 2);

    if solution > 10 {
        panic!("This is too big of a number...");
    }
}

fn main() {
    println!("We're working on hard problems today!");
    solve_hard_problem();
}