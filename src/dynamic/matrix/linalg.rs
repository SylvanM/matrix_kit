use algebra_kit::algebra::{Field, Ring};
use num_traits::Signed;
use crate::compatible_vectors;
use super::Matrix;

impl<R: Ring> Matrix<R> {

	// MARK: Math

	/// Computes the inner-product of this vector with another vector
	/// 
	/// This operates in the entries in the flatmap of each matrix, so if 
	/// the argument to this function are not proper vectors (i.e., they 
	/// are matrices with both dimensions greater than 1) then the behavior
	/// here is not well-defined
	pub fn inner_product(&self, other: &Matrix<R>) -> R {
		debug_assert_eq!(self.flatmap.len(), other.flatmap.len());

		let mut inner_product = R::zero();

		for i in 0..self.flatmap.len() {
			inner_product += self.flatmap[i].clone() * other.flatmap[i].clone();
		}

		inner_product
	}

	/// The squared L2 norm of this vector
	pub fn l2_norm_squared(&self) -> R {
		self.inner_product(self)
	}

	/// Computes the point-wise product of this and another matrix of the same 
	/// dimension
	pub fn hadamard(&self, other: Matrix<R>) -> Matrix<R> {

		debug_assert_eq!(self.col_count(), other.col_count());
		debug_assert_eq!(self.row_count(), other.row_count());

		let mut hada = self.clone();

		for i in 0..(self.flatmap.len()) { 
			hada.flatmap[i] *= other.flatmap[i].clone()
		}

		hada
	}

}

impl<R: Ring + Signed + PartialOrd> Matrix<R> {

	/// Computes the l-infinity norm of this vector 
	pub fn l_inf_norm(&self) -> R {
		// We do this by hand instead of calling max() since floats don't
		// conform to Ord, since they have NaN values.
		self.flatmap.iter()
			.map(|x| x.abs())
			.reduce(|a, b| if b > a { b } else { a })
			.unwrap()
	}

}

// MARK: Matrices over Fields

impl<F: Field> Matrix<F> {

	/// Returns whether or not two vectors are orthogonal
	pub fn is_orthogonal_to(&self, other: Matrix<F>) -> bool {
		debug_assert!(compatible_vectors!(self, other));
		self.inner_product(&other) == F::zero()
	}

	/// Returns whether or not a matrix is orthogonal (meaning its columns 
	/// are each orthogonal and of unit length)
	pub fn is_orthogonal(&self) -> bool {
		(self.transpose() * self.clone()).is_identity()
	}

	/// Projects this vector onto another vector
	pub fn proj_onto(&self, other: Matrix<F>) -> Matrix<F> {
		debug_assert!(compatible_vectors!(self, other));
		let scalar = other.inner_product(self) / other.inner_product(&other);
		other * scalar
	}

	/// Performs Gram-Schmidt Orthogonalization on a matrix, returning a
	/// matrix whose columns are orthogonal, and span the same column space 
	/// as the original matrix. This does NOT normalize the GS vectors.
	pub fn gram_schmidt(&self) -> Matrix<F> {
		let v = self.columns();

		let mut u = v.clone();

		// The first GS vector is just the normalized regular guy
		u[0] = v[0].clone();

		for k in 1..u.len() {
				u[k] = v[k].clone();
			for i in 0..k {
				u[k] = u[k].clone() - u[k].proj_onto(u[i].clone())
			}
		}

		Matrix::from_col_vectors(u)
	}

	

}

// MARK: Linear Algebra over the Reals

impl Matrix<f64> {

	// MARK: Vector utility

	/// Normalizes this vector
	pub fn normalize(&mut self) {
		debug_assert!(self.is_vector());
		*self /= self.l2_norm_squared().sqrt();
	}

	/// Returns the normalized version of this vector
	pub fn normalized(&self) -> Matrix<f64> {
		debug_assert!(self.is_vector());
		let mut unit = self.clone();
		unit.normalize();
		unit
	}

	/// Returns the angle of this 2D vector with the x-axis
	pub fn angle(&self) -> f64 {
		debug_assert!(self.is_vector());
		debug_assert_eq!(self.row_count(), 2);

		let x = self.get(0, 0);
		let y = self.get(1, 0);

		y.atan2(x)
	}

}
#[cfg(test)]
mod tests {
	use super::*;

}
