//
// Copyright (c) 2018-2021 INRIA
//

#ifndef __pinocchio_math_taylor_expansion_hpp__
#define __pinocchio_math_taylor_expansion_hpp__

#include "pinocchio/math/fwd.hpp"
#include <limits>

namespace pinocchio
{

  ///
  /// \brief Helper struct to retrieve some useful information for a Taylor series
  ///        expansion according to the a given Scalar type.
  ///
  /// \tparam Scalar the Scalar type of the Taylor series expansion.
  ///
  template<typename Scalar>
  struct TaylorSeriesExpansion
  {
    ///
    /// \brief Computes the expected tolerance of the argument of a Taylor series expansion for a
    /// certain degree
    ///        according to the machine precision of the given input Scalar.
    ///
    /// \tparam degree the degree of the Taylor series expansion.
    ///
    template<int degree>
    static Scalar precision()
    {
      constexpr Scalar a = std::numeric_limits<Scalar>::epsilon();
      constexpr Scalar b = Scalar(1) / Scalar(degree + 1);
      return math::pow(a, b);
    }

    static Scalar precision(const int degree)
    {
      constexpr Scalar a = std::numeric_limits<Scalar>::epsilon();
      const Scalar b = Scalar(1) / Scalar(degree + 1);
      return math::pow(a, b);
    }
  }; // struct TaylorSeriesExpansion

} // namespace pinocchio

#endif // ifndef __pinocchio_math_taylor_expansion_hpp__
