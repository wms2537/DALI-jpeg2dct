#include "jpeg2dct.h"
#include "dctfromjpg.h"

namespace jpeg2dct {

void JpegToDct::RunImpl(::dali::SampleWorkspace &ws) {
  const auto &input = ws.Input<::dali::CPUBackend>(0);
  auto &dct_y = ws.Output<::dali::CPUBackend>(0);
  auto &dct_cb = ws.Output<::dali::CPUBackend>(1);
  auto &dct_cr = ws.Output<::dali::CPUBackend>(2);
  auto file_name = input.GetSourceInfo();

  // Verify input
  DALI_ENFORCE(input.ndim() == 1,
               "Input must be 1D encoded jpeg string.");
  DALI_ENFORCE(::dali::IsType<::dali::uint8>(input.type()),
               "Input must be stored as uint8 data.");
  
  jpeg2dct::common::band_info bands[3];
  try {
    jpeg2dct::common::read_dct_coefficients_from_buffer_(
        (char*)input.data<::dali::uint8>(), input.size(), normalize_,
        channels_, &bands[0], &bands[1], &bands[2]);
  } catch (std::runtime_error &e) {
    DALI_FAIL(e.what() + ". File: " + file_name);
  }
  dct_y.set_type(::dali::DALIDataType::DALI_INT16);
  dct_y.Resize({bands[0].dct_h, bands[0].dct_w, bands[0].dct_b}, ::dali::DALIDataType::DALI_INT16);
  dct_y.SetLayout("HWC");
  std::memcpy(dct_y.mutable_data<short>(), bands[0].dct, sizeof(short) * bands[2].dct_h * bands[2].dct_w * bands[2].dct_b);
  delete[] bands[0].dct;

  if(channels_ > 1) {
    dct_cb.set_type(::dali::DALIDataType::DALI_INT16);
    dct_cb.Resize({bands[1].dct_h, bands[1].dct_w, bands[1].dct_b}, ::dali::DALIDataType::DALI_INT16);
    dct_cb.SetLayout("HWC");
    std::memcpy(dct_cb.mutable_data<short>(), bands[1].dct, sizeof(short) * bands[1].dct_h * bands[1].dct_w * bands[1].dct_b);
    delete[] bands[1].dct;
  }

  if(channels_ > 2) {
    dct_cr.set_type(::dali::DALIDataType::DALI_INT16);
    dct_cr.Resize({bands[2].dct_h, bands[2].dct_w, bands[2].dct_b}, ::dali::DALIDataType::DALI_INT16);
    dct_cr.SetLayout("HWC");
    std::memcpy(dct_cr.mutable_data<short>(), bands[2].dct, sizeof(short) * bands[2].dct_h * bands[2].dct_w * bands[2].dct_b);
    delete[] bands[2].dct;
  }
}

}  // namespace jpeg2dct

DALI_REGISTER_OPERATOR(JpegToDct, ::jpeg2dct::JpegToDct, ::dali::CPU);

DALI_SCHEMA(JpegToDct)
    .DocStr("Decode Jpeg string to dct coefficients")
    .NumInput(1)
    .NumOutput(3)
    .AddOptionalArg("normalize",
                    R"code(Normalize with quantification tables.)code",
                    true)
    .AddOptionalArg("channels",
                    R"code(Number of channels in input image.)code",
                    3);
