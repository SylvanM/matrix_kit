use super::Matrix;

impl<T: Clone> Matrix<T> {
	/// Iterates over all entries, in column-major order
	pub fn iter(&self) -> std::slice::Iter<'_, T> {
		self.flatmap.iter()
	}

	pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
		self.flatmap.iter_mut()
	}
}

impl<'a, T: Clone> IntoIterator for &'a Matrix<T> {
	type Item = &'a T;
	type IntoIter = std::slice::Iter<'a, T>;
	fn into_iter(self) -> Self::IntoIter { self.flatmap.iter() }
}

impl<T: Clone> IntoIterator for Matrix<T> {
	type Item = T;
	type IntoIter = std::vec::IntoIter<T>;
	fn into_iter(self) -> Self::IntoIter { self.flatmap.into_iter() }
}
