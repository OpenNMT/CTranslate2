#include "test_utils.h"
#include "ctranslate2/layers/layers.h"
#include "ctranslate2/padder.h"

class LayerDeviceFPTest : public ::testing::TestWithParam<FloatType> {
};

TEST(LayerTest, MakeRelativePositions1D) {
  const StorageView positions = layers::make_relative_positions(1, 4, 2);
  const StorageView expected({1, 4}, std::vector<int32_t>{0, 0, 1, 2});
  expect_storage_eq(positions, expected);
}

TEST(LayerTest, MakeRelativePositions2D) {
  const StorageView positions = layers::make_relative_positions(4, 4, 2);
  const StorageView expected({4, 4}, std::vector<int32_t>{
      2, 3, 4, 4,
      1, 2, 3, 4,
      0, 1, 2, 3,
      0, 0, 1, 2});
  expect_storage_eq(positions, expected);
}

TEST_P(LayerDeviceFPTest, Alibi) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = std::max(GetParam().error, float(1e-4));

  const StorageView zero({3, 4, 2, 5}, 0.f, device);

  {
    const StorageView expected({3, 4, 2, 5}, std::vector<float>{
        -1.0, -0.75, -0.5, -0.25, 0.0,
        -1.0, -0.75, -0.5, -0.25, 0.0,
        -0.25, -0.1875, -0.125, -0.0625, 0.0,
        -0.25, -0.1875, -0.125, -0.0625, 0.0,
        -0.0625, -0.046875, -0.03125, -0.015625, 0.0,
        -0.0625, -0.046875, -0.03125, -0.015625, 0.0,
        -0.015625, -0.01171875, -0.0078125, -0.00390625, 0.0,
        -0.015625, -0.01171875, -0.0078125, -0.00390625, 0.0,
        -1.0, -0.75, -0.5, -0.25, 0.0,
        -1.0, -0.75, -0.5, -0.25, 0.0,
        -0.25, -0.1875, -0.125, -0.0625, 0.0,
        -0.25, -0.1875, -0.125, -0.0625, 0.0,
        -0.0625, -0.046875, -0.03125, -0.015625, 0.0,
        -0.0625, -0.046875, -0.03125, -0.015625, 0.0,
        -0.015625, -0.01171875, -0.0078125, -0.00390625, 0.0,
        -0.015625, -0.01171875, -0.0078125, -0.00390625, 0.0,
        -1.0, -0.75, -0.5, -0.25, 0.0,
        -1.0, -0.75, -0.5, -0.25, 0.0,
        -0.25, -0.1875, -0.125, -0.0625, 0.0,
        -0.25, -0.1875, -0.125, -0.0625, 0.0,
        -0.0625, -0.046875, -0.03125, -0.015625, 0.0,
        -0.0625, -0.046875, -0.03125, -0.015625, 0.0,
        -0.015625, -0.01171875, -0.0078125, -0.00390625, 0.0,
        -0.015625, -0.01171875, -0.0078125, -0.00390625, 0.0});

    layers::Alibi alibi;
    StorageView x = zero.to(dtype);
    alibi.apply(x);
    expect_storage_eq(x.to_float32(), expected, error);
  }

  {
    const StorageView expected({3, 4, 2, 5}, std::vector<float>{
        0.0000, 0.2500, 0.5000, 0.7500, 1.0000,
        0.0000, 0.2500, 0.5000, 0.7500, 1.0000,
        0.0000, 0.0625, 0.1250, 0.1875, 0.2500,
        0.0000, 0.0625, 0.1250, 0.1875, 0.2500,
        0.0000, 0.0156, 0.0312, 0.0469, 0.0625,
        0.0000, 0.0156, 0.0312, 0.0469, 0.0625,
        0.0000, 0.0039, 0.0078, 0.0117, 0.0156,
        0.0000, 0.0039, 0.0078, 0.0117, 0.0156,
        0.0000, 0.2500, 0.5000, 0.7500, 1.0000,
        0.0000, 0.2500, 0.5000, 0.7500, 1.0000,
        0.0000, 0.0625, 0.1250, 0.1875, 0.2500,
        0.0000, 0.0625, 0.1250, 0.1875, 0.2500,
        0.0000, 0.0156, 0.0312, 0.0469, 0.0625,
        0.0000, 0.0156, 0.0312, 0.0469, 0.0625,
        0.0000, 0.0039, 0.0078, 0.0117, 0.0156,
        0.0000, 0.0039, 0.0078, 0.0117, 0.0156,
        0.0000, 0.2500, 0.5000, 0.7500, 1.0000,
        0.0000, 0.2500, 0.5000, 0.7500, 1.0000,
        0.0000, 0.0625, 0.1250, 0.1875, 0.2500,
        0.0000, 0.0625, 0.1250, 0.1875, 0.2500,
        0.0000, 0.0156, 0.0312, 0.0469, 0.0625,
        0.0000, 0.0156, 0.0312, 0.0469, 0.0625,
        0.0000, 0.0039, 0.0078, 0.0117, 0.0156,
        0.0000, 0.0039, 0.0078, 0.0117, 0.0156});

    layers::Alibi alibi(/*use_positive_positions=*/true);
    StorageView x = zero.to(dtype);
    alibi.apply(x);
    expect_storage_eq(x.to_float32(), expected, error);
  }
}

TEST_P(LayerDeviceFPTest, RotaryEmbedding) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;

  const StorageView input({2, 4, 2, 6}, std::vector<float>{
      0.8822692632675171, 0.9150039553642273, 0.38286375999450684, 0.9593056440353394,
      0.3904482126235962, 0.600895345211029, 0.10531491041183472, 0.26949483156204224,
      0.3588126301765442, 0.19936376810073853, 0.5471915602684021, 0.006160438060760498,
      0.2565724849700928, 0.7936413288116455, 0.9407714605331421, 0.13318592309951782,
      0.9345980882644653, 0.5935796499252319, 0.951554536819458, 0.07526588439941406,
      0.8860136866569519, 0.5832095742225647, 0.3376477360725403, 0.8089749813079834,
      0.8694044351577759, 0.5677152872085571, 0.7410940527915955, 0.42940449714660645,
      0.8854429125785828, 0.5739044547080994, 0.5779253840446472, 0.9039816856384277,
      0.5546598434448242, 0.34231340885162354, 0.634341835975647, 0.36441028118133545,
      0.2665800452232361, 0.6274491548538208, 0.26963168382644653, 0.4413635730743408,
      0.2969208359718323, 0.831685483455658, 0.710428774356842, 0.9464110732078552,
      0.7890297770500183, 0.281413733959198, 0.788632333278656, 0.5894631147384644,
      0.7539175152778625, 0.19524747133255005, 0.005045771598815918, 0.30681973695755005,
      0.11648857593536377, 0.9102694392204285, 0.9811253547668457, 0.08736187219619751,
      0.00406193733215332, 0.10881811380386353, 0.16365545988082886, 0.7025200724601746,
      0.6440156698226929, 0.7071067690849304, 0.6581305861473083, 0.4913020133972168,
      0.8913041353225708, 0.1447432041168213, 0.6790379285812378, 0.9154621958732605,
      0.24178731441497803, 0.1591441035270691, 0.7652890682220459, 0.2978977560997009,
      0.5314818620681763, 0.1587299108505249, 0.6541759967803955, 0.32780885696411133,
      0.6532081365585327, 0.3958292603492737, 0.8034619092941284, 0.38134968280792236,
      0.786022961139679, 0.11151599884033203, 0.2476751208305359, 0.652438223361969,
      0.9146959185600281, 0.20364904403686523, 0.20180100202560425, 0.20178300142288208,
      0.9497213959693909, 0.6666255593299866, 0.6057037711143494, 0.3725206255912781,
      0.7980347275733948, 0.8399046063423157, 0.13741332292556763, 0.2330659031867981
    }, device);

  const StorageView expected({2, 4, 2, 6}, std::vector<float>{
      -1.1991642713546753, 0.421469122171402, 0.29228904843330383, 0.9906659722328186,
      0.3878554105758667, 0.6025721430778503, -0.1422920823097229, -0.25193583965301514,
      0.3276682496070862, 0.24723657965660095, 0.5471403002738953, 0.009696949273347855,
      -0.8284278512001038, -0.09697063267230988, 0.9243745803833008, 0.2198205441236496,
      0.9320317506790161, 0.5976011753082275, -0.9526534080505371, 0.05977071821689606,
      0.7964892983436584, 0.7005414962768555, 0.33241209387779236, 0.8111404180526733,
      -0.8780219554901123, 0.5542942881584167, 0.6980978846549988, 0.4962538480758667,
      0.8829618096351624, 0.5777143239974976, -0.6997116804122925, -0.8133782148361206,
      0.5017786622047424, 0.41598576307296753, 0.6319733262062073, 0.36850258708000183,
      -0.6814743280410767, -0.01871044933795929, 0.227556973695755, 0.464457631111145,
      0.2933344542980194, 0.8329571485519409, -0.836876630783081, -0.8366841077804565,
      0.7423328161239624, 0.3882056474685669, 0.7848060131072998, 0.5945479869842529,
      -0.49127840995788574, 0.6042836308479309, -0.023417681455612183, 0.3059663772583008,
      0.11256527900695801, 0.9107629060745239, -0.9836352467536926, 0.05196882039308548,
      -0.01108112558722496, 0.10832861065864563, 0.15911148488521576, 0.7035631537437439,
      -0.910975456237793, 0.29134154319763184, 0.6097538471221924, 0.5501943230628967,
      0.8906721472740173, 0.1485823690891266, -0.8014324903488159, -0.8104748725891113,
      0.2173580825328827, 0.1911633163690567, 0.7633476853370667, 0.3028377890586853,
      -0.36550718545913696, 0.4172201156616211, 0.6209718585014343, 0.38703852891921997,
      0.6514964699745178, 0.3986401855945587, -0.849237322807312, -0.2641487717628479,
      0.7629365921020508, 0.21953508257865906, 0.24345307052135468, 0.6540254354476929,
      -0.565825343132019, 0.7469826936721802, 0.18222710490226746, 0.21962082386016846,
      0.9468401670455933, 0.6707115769386292, -0.6522123217582703, -0.2833157181739807,
      0.6737331748008728, 0.9425405859947205, 0.13590410351753235, 0.2339491844177246
    }, device);

  const auto permute = [](const StorageView& in) {
    StorageView x = in;
    x.reshape({8, 2, 3, 2});

    StorageView y(x.device());
    ops::Transpose({0, 1, 3, 2})(x, y);

    y.reshape({2, 4, 2, 6});
    return y;
  };

  {
    layers::RotaryEmbeddings rotary_embeddings;
    StorageView x = input.to(dtype);
    rotary_embeddings.apply(x, 2);
    expect_storage_eq(x.to_float32(), expected, error);
  }

  {
    layers::RotaryEmbeddings rotary_embeddings(0, false);
    StorageView x = permute(input).to(dtype);
    rotary_embeddings.apply(x, 2);
    expect_storage_eq(x.to_float32(), permute(expected), error);
  }
}

TEST(LayerTest, Padder) {
  const StorageView lengths({3}, std::vector<int32_t>{2, 3, 1});
  const Padder padder(lengths, /*max_time=*/4);

  StorageView x({3, 4}, std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  const StorageView wo_padding({6}, std::vector<int32_t>{0, 1, 4, 5, 6, 8});
  const StorageView w_padding({3, 4}, std::vector<int32_t>{0, 1, 1, 1, 4, 5, 6, 6, 8, 8, 8, 8});

  padder.remove_padding(x);
  ASSERT_EQ(x.rank(), 1);
  expect_storage_eq(x, wo_padding);
  padder.add_padding(x);
  ASSERT_EQ(x.rank(), 2);
  expect_storage_eq(x, w_padding);
}

TEST(LayerTest, PadderToMultiple) {
  const StorageView lengths({3}, std::vector<int32_t>{2, 3, 1});
  const Padder padder(lengths, /*max_time=*/4, /*pad_batch_to_multiple=*/8);

  StorageView x({3, 4}, std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  const StorageView wo_padding({8}, std::vector<int32_t>{0, 1, 4, 5, 6, 8, 8, 8});
  const StorageView w_padding({3, 4}, std::vector<int32_t>{0, 1, 1, 1, 4, 5, 6, 6, 8, 8, 8, 8});

  padder.remove_padding(x);
  expect_storage_eq(x, wo_padding);
  padder.add_padding(x);
  expect_storage_eq(x, w_padding);
}

TEST(LayerTest, PadderIgnore) {
  const StorageView lengths({3}, std::vector<int32_t>{4, 4, 4});
  const Padder padder(lengths);

  StorageView x({3, 4}, std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  const StorageView original(x);

  padder.remove_padding(x);
  expect_storage_eq(x, original);
  padder.add_padding(x);
  expect_storage_eq(x, original);
}

TEST(LayerTest, PositionEncoderNoSharedState) {
  // Test case for issue: http://forum.opennmt.net/t/ctranslate2-c-api-returns-strange-results-when-initializing-2-models/3208
  layers::SinusoidalPositionEncoder position_encoder_1(4);
  layers::SinusoidalPositionEncoder position_encoder_2(6);

  {
    StorageView input(
      {1, 1, 4}, std::vector<float>{0.1, -2.3, 0.5, 1.2});
    StorageView expected(
      {1, 1, 4}, std::vector<float>{0.941471, -2.2999, 1.0403, 2.2});
    position_encoder_1(input);
    expect_storage_eq(input, expected, 1e-5);
  }

  {
    StorageView input(
      {1, 1, 6}, std::vector<float>{-0.2, -1.3, 0.1, -0.6, 2.0, 1.1});
    StorageView expected(
      {1, 1, 6}, std::vector<float>{0.641471, -1.29, 0.1001, -0.0596977, 2.99995, 2.1});
    position_encoder_2(input);
    expect_storage_eq(input, expected, 1e-5);
  }
}


INSTANTIATE_TEST_SUITE_P(CPU, LayerDeviceFPTest,
                         ::testing::Values(FloatType{Device::CPU, DataType::FLOAT32, 1e-5}),
                         fp_test_name);
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, LayerDeviceFPTest,
                         ::testing::Values(FloatType{Device::CUDA, DataType::FLOAT32, 1e-5},
                                           FloatType{Device::CUDA, DataType::FLOAT16, 1e-2},
                                           FloatType{Device::CUDA, DataType::BFLOAT16, 1e-2}),
                         fp_test_name);
#endif
