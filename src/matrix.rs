//
// A matrix!
//

use std::{fmt::Debug, ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign}};

use crate::algebra::*;

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
/// 
/// * `M` - 
fn vec_dot_prod_ptr<const M: usize, const N: usize, R: Ring>(x: &[R], r: usize, y: &[R], out: &mut R) {
	*out = R::zero();
	for i in 0..N {
		*out += x[index!(M, N, r, i)] * y[i]
	}
}

fn mat_vec_mul_ptr<const M: usize, const N: usize, R: Ring>(a: &[R], vec: &[R], to_vec: &mut [R]) {
	for r in 0..M {
		vec_dot_prod_ptr::<M, N, R>(a, r, vec, &mut to_vec[r]);
	}
}

fn mat_add<const M: usize, const N: usize, R: Ring>(a: &[R], b: &[R], out: &mut [R]) {
	for i in 0..(M * N) {
		out[i] = a[i] + b[i];
	}
}

fn scalar_mul<const N: usize, R: Ring>(k: R, v: &[R], out: &mut [R]) {
	for i in 0..N {
		out[i] = k * v[i];
	}
}

fn mat_mul_ptrs<const M: usize, const K: usize, const N: usize, R: Ring>(a: &[R], b: &[R], out: &mut [R]) {
	for c in 0..N {
		mat_vec_mul_ptr::<M, K, R>(a, 
			&b[index!(K, N, 0, c)..index!(K, N, K, c)], 
			&mut out[index!(M, N, 0, c)..index!(M, N, M, c)]
		);
	}
}

#[cfg(test)]
mod tests {
    use super::*;

	#[test]
	fn test_mat_vec_mul_ptr() {
		// test the super simple identity!
		let identity = [1.into(), 0.into(), 0.into(), 1.into()];
		let simple_vector = [3.into(), 7.into()];
		let mut out_vector = [0.into() ; 2];
		mat_vec_mul_ptr::<2, 2, ZM<11>>(&identity, &simple_vector, &mut out_vector);

		assert_eq!(out_vector, simple_vector);

		let mat = [3.into(), 2.into(), 5.into(), 1.into(), 7.into(), 0.into()];
		let vec = [1.into(), 4.into(), 9.into()];
		let mut out = [0.into() ; 2];
		mat_vec_mul_ptr::<2, 3, ZM<11>>(&mat, &vec, &mut out);

		assert_eq!(out, [9.into(), 6.into()]);

	}

	#[test]
	fn test_full_mat_mul() {
		let a = [4.into(), 8.into(), 5.into(), 5.into(), 6.into(), 5.into(), 9.into(), 1.into(), 10.into(), 2.into(), 1.into(), 0.into()];
		let b = [1.into(), 2.into(), 8.into(), 3.into(), 1.into(), 4.into(), 0.into(), 7.into()];

		let mut prod = [0.into() ; 3 * 2];
		mat_mul_ptrs::<3, 4, 2, ZM<11>>(&a, &b, &mut prod);
		

		assert_eq!(prod, [4.into(), 9.into(), 7.into(), 5.into(), 6.into(), 3.into()]);
	}
}

#[derive(Clone, Copy)]
pub struct Matrix<const M: usize, const N: usize, R: Ring> where [R; M * N]: Sized  {
	pub flatmap: [R ; M * N]
}

impl<const M: usize, const N: usize, R: Ring> Matrix<M, N, R> where [(); M * N]: Sized  {

	pub fn new() -> Self {
		Matrix::from_flatmap([R::zero() ; M * N])
	}

	pub fn from_flatmap(flatmap: [R ; M * N]) -> Self {
		Matrix { flatmap }
	} 

	pub fn identity() -> Self {
		let mut mat = Matrix::new();

		for r in 0..N {
			for c in 0..N {
				if r == c {
					mat[r][c] = R::one();
				}
			}
		}

		mat
	}

	pub fn rhs_concat<const K: usize>(self, rhs: Matrix<M, K, R>) -> Matrix<M, {N + K}, R> where [() ; M * K]: Sized, [() ; M * {N + K}]: Sized {
		let mut big_mat = Matrix::<M, {N + K}, R>::new();

		for r in 0..M {
			for c in 0..N {
				big_mat.flatmap[index!(M, N + K, r, c)] = self.flatmap[index!(M, N, r, c)]
			}
		}

		for r in 0..M {
			for c in 0..K {
				big_mat.flatmap[index!(M, N + K, r, c + N)] = rhs.flatmap[index!(M, K, r, c)]
			}
		}
		
		big_mat
	}

}

impl<const M: usize, const N: usize, R: Ring> Index<usize> for Matrix<M, N, R> where [() ; M * N]: Sized {
	type Output = [R];

	/**
	 * Indexes a single COLUMN of this matrix
	 */
	fn index(&self, index: usize) -> &Self::Output {
		println!("Indexing a {:?} * {:?} matrix at row {:?}, this goes from raw index {:?} to raw address {:?}", M, N, index, index!(M, N, index, 0), index!(M, N, index, N));
		&self.flatmap[index!(M, N, 0, index)..index!(M, N, M, index)]
	}
}

impl<const M: usize, const N: usize, R: Ring> IndexMut<usize> for Matrix<M, N, R> where [() ; M * N]: Sized {
	
	/**
	 * Indexes a single row of this matrix
	 */
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.flatmap[index!(M, N, 0, index)..index!(M, N, M, index)]
	}
}

impl<const M: usize, const N: usize, R: Ring> Add<Matrix<M, N, R>> for Matrix<M, N, R> where [R ; M * N]: Sized {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		let mut sum = Matrix::new();
		mat_add::<M, N, R>(&self.flatmap, &rhs.flatmap, &mut sum.flatmap);
		sum
	}
}

impl<const M: usize, const N: usize, R: Ring> AddAssign<Matrix<M, N, R>> for Matrix<M, N, R> where [R ; M * N]: Sized {
	fn add_assign(&mut self, rhs: Self) {
		for i in 0..(M * N) {
			self.flatmap[i] += rhs.flatmap[i]
		}
	}
}

impl<const M: usize, const N: usize, R: Ring> Sub<Matrix<M, N, R>> for Matrix<M, N, R> where [R ; M * N]: Sized {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output {
		let mut sum = Matrix::new();

		for i in 0..(M * N) {
			sum.flatmap[i] = self.flatmap[i] - rhs.flatmap[i]
		}

		sum
	}
}

impl<const M: usize, const N: usize, R: Ring> SubAssign<Matrix<M, N, R>> for Matrix<M, N, R> where [R ; M * N]: Sized {
	fn sub_assign(&mut self, rhs: Self) {
		for i in 0..(M * N) {
			self.flatmap[i] -= rhs.flatmap[i]
		}
	}
}

impl<const M: usize, const K: usize, const N: usize, R: Ring> Mul<Matrix<K, N, R>> for Matrix<M, K, R> where [R ; M * K]: Sized, [R ; K * N]: Sized, [R ; M * N]: Sized,  {
	type Output = Matrix<M, N, R>;

	fn mul(self, rhs: Matrix<K, N, R>) -> Self::Output {
		println!("Multiplying {:?} by {:?}", self, rhs);

		let mut prod = Matrix::<M, N, R>::new();
		mat_mul_ptrs::<M, K, N, R>(&self.flatmap, &rhs.flatmap, &mut prod.flatmap);
		prod
	}
}

impl<const M: usize, const N: usize, R: Ring> Neg for Matrix<M, N, R> where [R ; M * N]: Sized, <R as Neg>::Output: Ring {
	type Output = Self;

	fn neg(self) -> Self::Output {
		let mut negative = Matrix::new();
		scalar_mul::<{M * N}, R>(-R::one(), &self.flatmap, &mut negative.flatmap);
		negative
	}
}

impl<const M: usize, const N: usize, R: Ring> PartialEq for Matrix<M, N, R> where [R ; M * N]: Sized {
	fn eq(&self, other: &Self) -> bool {
		for i in 0..(M * N) {
			if self.flatmap[i] != other.flatmap[i] {
				return false;
			}
		}
		return true;
	}

	fn ne(&self, other: &Self) -> bool {
		for i in 0..(M * N) {
			if self.flatmap[i] == other.flatmap[i] {
				return true;
			}
		}
		return false;
	}
}

impl<const M: usize, const N: usize, R: Ring> Debug for Matrix<M, N, R> where [R ; M * N]: Sized {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		// get the widest value so we know how many spaces we need!
		let mut widest_str_len = 0;
		for i in 0..(M * N) {
			let as_str = format!("{:?}", self.flatmap[i]);
			let this_len = as_str.chars().count();
			if this_len > widest_str_len {
				widest_str_len = this_len;
			}
		}

		// now, we make a vector of strings!
		let mut lines = Vec::<String>::new();

		for r in 0..M {
			let mut this_line = Vec::<String>::new();

			if r == 0 {
				this_line.push("┌ ".to_string());
			} else if r == M - 1 {
				this_line.push("└ ".to_string())
			} else {
				this_line.push("│ ".to_string())
			}

			for c in 0..N {
				let this_entry_str = format!("{:?}", self.flatmap[index!(M, N, r, c)]);
				let this_entry_len = this_entry_str.chars().count();
				this_line.push(format!("{}{}", this_entry_str, " ".repeat(widest_str_len - this_entry_len + 1)));
			}

			if r == 0 {
				this_line.push("┐ ".to_string());
			} else if r == M - 1 {
				this_line.push("┘ ".to_string())
			} else {
				this_line.push("│ ".to_string())
			}

			lines.push(this_line.join(""));
		}

		write!(f, "\n{}", lines.join("\n"))
	}
}