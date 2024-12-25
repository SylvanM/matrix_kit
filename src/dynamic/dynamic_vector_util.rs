use algebra_kit::algebra::*;

/**
 * Translates a row-column coordinate to a flat index in the one dimensional array,
 * indexing along columns. (So, columns are stored next to each other in memory instead of rows)
 */
#[macro_export]
macro_rules! index {
	($m: expr, $n: expr, $r: expr, $c: expr) => {
		$c * $m + $r
	};
}

/// Computes the dot product between the r-th row of the matrix x, and the vector y.
pub fn vec_dot_prod_ptr<R: Ring>(m: usize, n: usize, x: &[R], r: usize, y: &[R], out: &mut R) {
	*out = R::zero();
	for i in 0..n {
		*out += x[index!(m, n, r, i)].clone() * y[i].clone()
	}
}

/// Computes a matrix product, and write the output to a vector pointer
pub fn mat_vec_mul_ptr<R: Ring>(m: usize, n: usize, a: &[R], vec: &[R], to_vec: &mut [R]) {
	for r in 0..m {
		vec_dot_prod_ptr(m, n, a, r, vec, &mut to_vec[r]);
	}
}

/// Adds a matrix a to a matrix b
pub fn mat_add<R: Ring>(m: usize, n: usize, a: &[R], b: &[R], out: &mut [R]) {
	for i in 0..(m * n) {
		out[i] = a[i].clone() + b[i].clone();
	}
}

/// Adds a matrix b to a, in place
pub fn mat_add_assign<R: Ring>(m: usize, n: usize, a: &mut [R], b: &[R]) {
	for i in 0..(m * n) {
		a[i] += a[i].clone() + b[i].clone();
	}
}

/// Subtracts a matrix b from a matrix a
pub fn mat_sub<R: Ring>(m: usize, n: usize, a: &[R], b: &[R], out: &mut [R]) {
	for i in 0..(m * n) {
		out[i] = a[i].clone() - b[i].clone();
	}
}

/// Subtracts a matrix b from a, in place
pub fn mat_sub_assign<R: Ring>(m: usize, n: usize, a: &mut [R], b: &[R]) {
	for i in 0..(m * n) {
		a[i] -= b[i].clone();
	}
}

/// Scales a vector v by k
pub fn scalar_mul<R: Ring>(n: usize, k: R, v: &[R], out: &mut [R]) {
	for i in 0..n {
		out[i] = k.clone() * v[i].clone();
	}
}

/// Scales a vector v by k, in place
pub fn scalar_mul_assign<R: Ring>(n: usize, k: R, v: &mut [R]) {
	for i in 0..n {
		v[i] *= k.clone();
	}
}

/// Computes the product of an m*k matrix and a k*n matrix
pub fn mat_mul_ptrs<R: Ring>(m: usize, k: usize, n: usize, a: &[R], b: &[R], out: &mut [R]) {
	for c in 0..n {
		mat_vec_mul_ptr::<R>(m, k, a,
			&b[index!(k, n, 0, c)..index!(k, n, k, c)],
			&mut out[index!(m, n, 0, c)..index!(m, n, m, c)]
		);
	}
}

/// Computes the product of two n*n matrices, in place. (This only works on squares!)
pub fn mat_mul_ptrs_assign<R: Ring>(n: usize, a: &mut [R], b: &[R]) {
	
	// There's probably some really cool, memory-efficient way to do this.
	// Unfortunately I haven't yet come across it (other than doing it 
	// row by row, which comes with its own setbacks)

	let mut out = vec![R::zero() ; n * n];
	mat_mul_ptrs(n, n, n, a, b, &mut out);
	
	for i in 0..(n * n) {
		a[i] = out[i].clone();
	}
}

#[cfg(test)]
mod tests {

    use super::*;

	#[test]
	fn test_mat_vec_mul_ptr() {

		// test the super simple identity!
		let identity = [1, 0, 0, 1];
		let simple_vector = [3, 7];
		let mut out_vector = [0 ; 2];
		mat_vec_mul_ptr(2, 2, &identity, &simple_vector, &mut out_vector);

		assert_eq!(out_vector, simple_vector);

		// let mat = [3, 2, 5, 1, 7, 0];
		let mat = [3, 1, 2, 7, 5, 0];
		let vec = [1, 4, 9];
		let mut out = [0 ; 2];
		mat_vec_mul_ptr(2, 3, &mat, &vec, &mut out);

		assert_eq!(out, [56, 29]);

	}

	#[test]
	fn test_full_mat_mul() {
		let a = [4, 6, 10, 8, 5, 2, 5, 9, 1, 5, 1, 0];
		let b = [1, 8, 1, 0, 2, 3, 4, 7];

		let mut prod = [0 ; 3 * 2];
		mat_mul_ptrs(3, 4, 2, &a, &b, &mut prod);
		

		assert_eq!(prod, [73, 55, 27, 87, 70, 30]);
	}
}