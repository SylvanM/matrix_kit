//
// A collection of primitive algebraic structures
//

use std::fmt::Debug;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ops::{Add, Neg, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

// MARK: Ring

/**
 * An algebraic ring
 */
pub trait Ring: Debug + PartialEq + Copy + Sized + Add<Self> + AddAssign<Self> + Neg + Sub<Self> + SubAssign<Self> + Mul<Self> + MulAssign<Self> + Mul<Output = Self> + Add<Output = Self> + Neg<Output = Self> + Sub<Output = Self> {

	/**
	 * The multiplicative identity of this ring
	 */
	fn one() -> Self;

	/**
	 * The additive identity of this ring
	 */
	 fn zero() -> Self;

	/**
	 * A ring element raised to a power
	 */
   	fn power(self, n: i64) -> Self;
}

// MARK: Field

/**
 * A field, which is a ring where every nonzero element has a multiplicative inverse
 */
pub trait Field: Ring + Div + DivAssign + Div<Output = Self> {
	fn inverse(self) -> Self;
}

// MARK: Inner Product Space
pub trait InnerProductSpace<R: Ring> {
	fn inner_product(self, other: Self) -> R;
}

pub trait NormSpace {
	fn norm(self) -> f64;
}

// MARK: Implementations

impl Ring for f64 {
	fn one() -> Self {
		1.0
	}

	fn zero() -> Self {
		0.0
	}

	fn power(self, n: i64) -> Self {
		self.powf(n as f64)
	}
}

impl Field for f64 {
	fn inverse(self) -> Self {
		if self == 0.0 {
			panic!("Cannot divide by zero")
		} else {
			1.0 / self
		}
	}
}

impl Ring for f32 {

	fn one() -> Self {
		1.0
	}

	fn zero() -> Self {
		0.0
	}

	fn power(self, n: i64) -> Self {
		self.powf(n as f32)
	}
}

impl Field for f32 {
	fn inverse(self) -> Self {
		if self == 0.0 {
			panic!("Cannot divide by zero")
		} else {
			1.0 / self
		}
	}
}

impl Ring for i8 {
	fn one() -> Self {
		1
	}

	fn zero() -> Self {
		0
	}

	fn power(self, n: i64) -> Self {
		if n < 0 {
			panic!("Cannot invert ring element")
		}
		self.pow(n as u32)
	}
}

impl Ring for i16 {
	fn one() -> Self {
		1
	}

	fn zero() -> Self {
		0
	}

	fn power(self, n: i64) -> Self {
		if n < 0 {
			panic!("Cannot invert ring element")
		}
		self.pow(n as u32)
	}
}

impl Ring for i32 {
	fn one() -> Self {
		1
	}

	fn zero() -> Self {
		0
	}

	fn power(self, n: i64) -> Self {
		if n < 0 {
			panic!("Cannot invert ring element")
		}
		self.pow(n as u32)
	}
}

impl Ring for i64 {
	fn one() -> Self {
		1
	}

	fn zero() -> Self {
		0
	}

	fn power(self, n: i64) -> Self {
		if n < 0 {
			panic!("Cannot invert ring element")
		}
		self.pow(n as u32)
	}
}

impl Ring for i128 {
	fn one() -> Self {
		1
	}

	fn zero() -> Self {
		0
	}

	fn power(self, n: i64) -> Self {
		if n < 0 {
			panic!("Cannot invert ring element")
		}
		self.pow(n as u32)
	}
}

/// The field of the integers modulo a prime Q
#[derive(Clone, Copy, Default)]
pub struct ZM<const Q: i64> {
	pub val: i64
}

impl<const Q: i64> ZM<Q> {
	pub fn rnd() -> ZM<Q> {
		ZM::<Q> { val: StdRng::from_entropy().gen::<i64>().rem_euclid(Q) }
	}
}

impl<const Q: i64> Debug for ZM<Q> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		self.val.fmt(f)
	}
}

impl<const Q: i64> PartialEq for ZM<Q> {
	fn eq(&self, other: &Self) -> bool {
		self.val == other.val
	}

	fn ne(&self, other: &Self) -> bool {
		self.val != other.val
	}
}

impl<const Q: i64> ZM<Q> {
	pub fn convert<const P: i64>(other: ZM<P>) -> ZM<Q> {
		other.val.into()
	}

	pub fn from_int(x: i64) -> ZM<Q> {
		x.into()
	}
}

impl<const Q: i64> From<i64> for ZM<Q> {
	fn from(value: i64) -> Self {
		ZM::<Q> { val: value.rem_euclid(Q) } 
	}
}

impl<const Q: i64> From<u8> for ZM<Q> {
	fn from(value: u8) -> Self {
		ZM::<Q> { val: (value as i64).rem_euclid(Q) } 
	}
}

impl<const Q: i64> From<i32> for ZM<Q> {
	fn from(value: i32) -> Self {
		ZM::<Q> { val: (value as i64).rem_euclid(Q) } 
	}
}

impl<const Q: i64> Add<ZM<Q>> for ZM<Q> {
	type Output = ZM<Q>;

	fn add(self, rhs: ZM<Q>) -> Self::Output {
		ZM::<Q> { val: (self.val + rhs.val) % Q }
	}
}

impl<const Q: i64> AddAssign<ZM<Q>> for ZM<Q> {
	fn add_assign(&mut self, rhs: ZM<Q>) {
		*self = *self + rhs
	}
}

impl<const Q: i64> Sub<ZM<Q>> for ZM<Q> {
	type Output = ZM<Q>;

	fn sub(self, rhs: ZM<Q>) -> Self::Output {
		ZM::<Q> { val: (self.val - rhs.val + Q) % Q }
	}
}

impl<const Q: i64> SubAssign<ZM<Q>> for ZM<Q> {
	fn sub_assign(&mut self, rhs: ZM<Q>) {
		*self = *self - rhs
	}
}

impl<const Q: i64> Mul<ZM<Q>> for ZM<Q> {
	type Output = ZM<Q>;

	fn mul(self, rhs: ZM<Q>) -> ZM<Q> {
		let product = self.val.rem_euclid(Q) * rhs.val.rem_euclid(Q);
		ZM::<Q> { val: product % Q }
	}
}

impl<const Q: i64> MulAssign<ZM<Q>> for ZM<Q> {
	fn mul_assign(&mut self, rhs: ZM<Q>) {
		*self = *self * rhs
	}
}

impl<const Q: i64> Neg for ZM<Q> {
	type Output = Self;

	fn neg(self) -> Self::Output {
		(Q - self.val).into()
	}
}

impl<const Q: i64> Ring for ZM<Q> {
	fn one() -> Self {
		ZM::<Q> { val: 1 }
	}

	fn zero() -> Self {
		ZM::<Q> { val: 0 }
	}

	fn power(self, n: i64) -> Self {
		// TODO: Make this WAYY more efficient... Double and add, yeah?
		let mut power = ZM::<Q>::one();

		for _ in 1..=n {
			power *= self
		}

		power
	}
}

// impl<const Q: i64> ZM<Q> {
// 	fn rnd() -> ZM<Q> {
// 		ZM::<Q> { val: StdRng::from_entropy().gen::<i64>().rem_euclid(Q) }
// 	}
// }

/// Returns (g, x, y) so that 
/// - g = gcd(a, b)
/// ax + by = gcd(a, b)
fn ext_gcd(a: i64, b: i64) -> (i64, i64, i64) {

	if a == 0 {
		return (b, 0, 1)
	}

	let (g, x1, y1) = ext_gcd(b % a, a);

	let x = y1 - (b/a) * x1;
	let y = x1;

	(g, x, y)
}

fn mod_inv(x: i64, m: i64) -> i64 {
	match ext_gcd(x, m) { (_, i, _) => i }
}

impl<const Q: i64> Div<ZM<Q>> for ZM<Q> {
	type Output = ZM<Q>;

	fn div(self, rhs: ZM<Q>) -> Self::Output {
		self * mod_inv(rhs.val, Q).into()
	}
}

impl<const Q: i64> DivAssign<ZM<Q>> for ZM<Q> {
	fn div_assign(&mut self, rhs: ZM<Q>) {
		*self = *self / rhs
	}
}

impl<const Q: i64> Field for ZM<Q> {
	fn inverse(self) -> Self {
		mod_inv(self.val, Q).into()
	}
}
