//
// Copyright (c) 2016-2022 CNRS INRIA
//

#include <boost/fusion/container/generation/make_list.hpp>
#include <pinocchio/multibody/model.hpp>
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/sample-models.hpp"
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/check.hpp>
#include <pinocchio/algorithm/default-check.hpp>
#include <iostream>

using namespace pinocchio;

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

// Dummy checker.
struct Check1 : public AlgorithmCheckerBase<Check1>
{
  bool checkModel_impl(const Model &) const
  {
    return true;
  }
};

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_check)
{
  using namespace boost::fusion;

  pinocchio::Model model;
  buildModels::humanoidRandom(model);

  BOOST_CHECK(model.check(Check1()));
  BOOST_CHECK(model.check(CRBAChecker()));
  BOOST_CHECK(model.check(ABAChecker()));

  BOOST_CHECK(model.check(makeAlgoCheckerList(Check1(), ParentChecker(), CRBAChecker())));
  BOOST_CHECK(model.check(DEFAULT_CHECKERS));

  pinocchio::Data data(model);
  BOOST_CHECK(checkData(model, data));
  BOOST_CHECK(model.check(data));

  BOOST_FOREACH (Inertia & Y, model.inertias)
  {
    Y.inertia().data().fill(-1.);
  }
  BOOST_CHECK(!model.check(ABAChecker())); // some inertias are negative ... check fail.

  pinocchio::Model model_mimic;
  const bool using_free_flyer = true;
  const bool using_mimic = true;
  buildModels::humanoidRandom(model_mimic, using_free_flyer, using_mimic);
  pinocchio::Data data_mimic(model_mimic);
  BOOST_CHECK(model_mimic.check(MimicChecker()) == false);
  BOOST_CHECK(model_mimic.check(data_mimic));
}

BOOST_AUTO_TEST_CASE(test_mimic_check)
{
  {
    Model manipulator_no_mimic;
    const bool using_mimic = false;
    buildModels::manipulator(manipulator_no_mimic, using_mimic);
    BOOST_CHECK(manipulator_no_mimic.mimicked_joints.size() == 0);
    BOOST_CHECK(manipulator_no_mimic.mimicking_joints.size() == 0);
    BOOST_CHECK(manipulator_no_mimic.check(MimicChecker()) == true);
  }
  {
    Model manipulator_mimic;
    const bool using_mimic = true;
    buildModels::manipulator(manipulator_mimic, using_mimic);
    BOOST_CHECK(manipulator_mimic.mimicked_joints.size() > 0);
    BOOST_CHECK(manipulator_mimic.mimicking_joints.size() > 0);
    BOOST_CHECK(manipulator_mimic.check(MimicChecker()) == false);
  }
  {
    Model humanoid_ff_no_mimic;
    const bool using_free_flyer = true;
    const bool using_mimic = false;
    buildModels::humanoidRandom(humanoid_ff_no_mimic, using_free_flyer, using_mimic);
    BOOST_CHECK(humanoid_ff_no_mimic.mimicked_joints.size() == 0);
    BOOST_CHECK(humanoid_ff_no_mimic.mimicking_joints.size() == 0);
    BOOST_CHECK(humanoid_ff_no_mimic.check(MimicChecker()) == true);
  }
  {
    Model humanoid_ff_mimic;
    const bool using_free_flyer = true;
    const bool using_mimic = true;
    buildModels::humanoidRandom(humanoid_ff_mimic, using_free_flyer, using_mimic);
    BOOST_CHECK(humanoid_ff_mimic.mimicked_joints.size() > 0);
    BOOST_CHECK(humanoid_ff_mimic.mimicking_joints.size() > 0);
    BOOST_CHECK(humanoid_ff_mimic.check(MimicChecker()) == false);
  }
  {
    Model humanoid_no_ff_no_mimic;
    const bool using_free_flyer = false;
    const bool using_mimic = false;
    buildModels::humanoidRandom(humanoid_no_ff_no_mimic, using_free_flyer, using_mimic);
    BOOST_CHECK(humanoid_no_ff_no_mimic.mimicked_joints.size() == 0);
    BOOST_CHECK(humanoid_no_ff_no_mimic.mimicking_joints.size() == 0);
    BOOST_CHECK(humanoid_no_ff_no_mimic.check(MimicChecker()) == true);
  }
  {
    Model humanoid_no_ff_mimic;
    const bool using_free_flyer = false;
    const bool using_mimic = true;
    buildModels::humanoidRandom(humanoid_no_ff_mimic, using_free_flyer, using_mimic);
    BOOST_CHECK(humanoid_no_ff_mimic.mimicked_joints.size() > 0);
    BOOST_CHECK(humanoid_no_ff_mimic.mimicking_joints.size() > 0);
    BOOST_CHECK(humanoid_no_ff_mimic.check(MimicChecker()) == false);
  }
  {
    Model custom_model;
    custom_model.addJoint(0, JointModelRX(), SE3::Identity(), "j1");
    custom_model.addJoint(1, JointModelRX(), SE3::Identity(), "j2");
    custom_model.addJoint(
      2, JointModelMimic(JointModelRX(), custom_model.joints[1], 1., 0.), SE3::Identity(), "j3");
    BOOST_CHECK(custom_model.mimicked_joints.size() > 0);
    BOOST_CHECK(custom_model.mimicking_joints.size() > 0);
    BOOST_CHECK(custom_model.check(MimicChecker()) == false);
  }
}

BOOST_AUTO_TEST_SUITE_END()
