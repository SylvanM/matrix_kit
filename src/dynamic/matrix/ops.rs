use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use algebra_kit::algebra::{Field, Ring};
use crate::index;
use crate::dynamic::dynamic_vector_util::*;
use super::Matrix;

// MARK: Index

impl<T: Clone> Index<usize> for Matrix<T> {
	type Output = [T];

	/// Returns the column `index` as a slice
	fn index(&self, index: usize) -> &Self::Output {
		&self.flatmap[
			index!(self.row_count, self.col_count, 0, index)..
			index!(self.row_count, self.col_count, self.row_count, index)
		]
	}
}

impl<T: Clone> IndexMut<usize> for Matrix<T> {

	/// Returns a mutable reference to the column `index`
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.flatmap[
			index!(self.row_count, self.col_count, 0, index)..
			index!(self.row_count, self.col_count, self.row_count, index)
		]
	}
}

// MARK: Comparison

impl<T: Clone + PartialEq> PartialEq for Matrix<T> {
	fn eq(&self, other: &Self) -> bool {
		self.row_count == other.row_count && self.col_count == other.col_count && self.flatmap == other.flatmap
	}
}


// MARK: Operations

impl<R: Ring> Add for Matrix<R> {
	type Output = Matrix<R>;

	fn add(self, rhs: Matrix<R>) -> Matrix<R> {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		let mut out = Matrix::<R>::new(self.row_count, self.col_count);
		mat_add(self.row_count, self.col_count, 
			&self.flatmap, &rhs.flatmap,
			&mut out.flatmap
		);
		out
	}
}

impl<R: Ring> AddAssign for Matrix<R> {
	fn add_assign(&mut self, rhs: Matrix<R>) {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		mat_add_assign(self.row_count, self.col_count, 
			&mut self.flatmap, &rhs.flatmap
		);
	}
}

impl<R: Ring> Sub for Matrix<R> {
	type Output = Matrix<R>;

	fn sub(self, rhs: Matrix<R>) -> Matrix<R> {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		let mut out = Matrix::<R>::new(self.row_count, self.col_count);
		mat_sub(self.row_count, self.col_count, 
			&self.flatmap, &rhs.flatmap, 
			&mut out.flatmap
		);
		out
	}
}

impl<R: Ring> SubAssign for Matrix<R> {
	fn sub_assign(&mut self, rhs: Matrix<R>) {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		mat_sub_assign(self.row_count, self.col_count, 
			&mut self.flatmap, &rhs.flatmap
		);
	}
}

impl<R: Ring> Mul<R> for Matrix<R> {
	type Output = Matrix<R>;

	fn mul(self, rhs: R) -> Self::Output {
		let mut out = Matrix::<R>::new(self.row_count, self.col_count);
		scalar_mul(self.row_count * self.col_count, 
			rhs, 
			&self.flatmap, &mut out.flatmap
		);
		out
	}
}

impl<R: Ring> MulAssign<R> for Matrix<R> {
	fn mul_assign(&mut self, rhs: R) {
		scalar_mul_assign(self.row_count * self.col_count, 
			rhs, &mut self.flatmap
		);
	}
}

impl<F: Field> Div<F> for Matrix<F> {
	type Output = Matrix<F>;

	fn div(self, rhs: F) -> Matrix<F> {
		let mut out = Matrix::<F>::new(self.row_count, self.col_count);
		scalar_div(self.row_count * self.col_count, 
			rhs, 
			&self.flatmap, &mut out.flatmap
		);
		out
	}
}

impl<F: Field> DivAssign<F> for Matrix<F> {
	fn div_assign(&mut self, rhs: F) {
		scalar_div_assign(self.row_count * self.col_count, 
			rhs, &mut self.flatmap
		);
	}
}

impl<R: Ring> Mul<Matrix<R>> for Matrix<R> {
	type Output = Matrix<R>;

	fn mul(self, rhs: Matrix<R>) -> Self::Output {
		assert_eq!(self.col_count, rhs.row_count, "Attempting to multiply a {} x {} matrix with a {} x {} matrix.", self.row_count(), self.col_count(), rhs.row_count(), rhs.col_count());

		let mut out = Matrix::<R>::new(self.row_count, rhs.col_count);
		mat_mul_ptrs::<R>(self.row_count, self.col_count, rhs.col_count, 
			&self.flatmap, &rhs.flatmap, &mut out.flatmap
		);
		out
	}
}

impl<R: Ring> MulAssign<Matrix<R>> for Matrix<R> {

	/// Performs in-place multiplication, only valid on square matrices
	fn mul_assign(&mut self, rhs: Matrix<R>) {
		assert_eq!(self.col_count, self.row_count); // Only on square matrices!
		assert_eq!(self.col_count, rhs.col_count); // make sure the other matrix is chill
		assert_eq!(self.row_count, rhs.row_count);

		mat_mul_ptrs_assign(self.row_count, 
			&mut self.flatmap, &rhs.flatmap
		);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_add() {
		let a = Matrix::from_flatmap(2, 2, vec![1, 2, 3, 4]);
		let b = Matrix::from_flatmap(2, 2, vec![5, 6, 7, 8]);
		let c = a + b;
		assert_eq!(c.flatmap, vec![6, 8, 10, 12]);
	}

}
