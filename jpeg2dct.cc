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
  ::dali::Image::Shape dct_y_shape = {bands[0].dct_h, bands[0].dct_w, bands[0].dct_b};
  dct_y.Resize(dct_y_shape, ::dali::TypeTable::GetTypeInfo(::dali::DALIDataType::DALI_INT16));
  dct_y.SetLayout("HWC");
  unsigned char *out_data = dct_y.mutable_data<unsigned char>();
  std::memcpy(out_data, (void *)(bands[0].dct), ::dali::volume(dct_y_shape));
  delete[] bands[0].dct;

  if(channels_ > 1) {
    ::dali::Image::Shape dct_cb_shape = {bands[1].dct_h, bands[1].dct_w, bands[1].dct_b};
    dct_cb.Resize(dct_cb_shape, ::dali::TypeTable::GetTypeInfo(::dali::DALIDataType::DALI_INT16));
    dct_cb.SetLayout("HWC");
    unsigned char *out_data = dct_cb.mutable_data<unsigned char>();
    std::memcpy(out_data, (void *)(bands[1].dct), ::dali::volume(dct_cb_shape));
    delete[] bands[1].dct;
  }

  if(channels_ > 2) {
    ::dali::Image::Shape dct_cr_shape = {bands[2].dct_h, bands[2].dct_w, bands[2].dct_b};
    dct_cr.Resize(dct_cr_shape, ::dali::TypeTable::GetTypeInfo(::dali::DALIDataType::DALI_INT16));
    dct_cr.SetLayout("HWC");
    unsigned char *out_data = dct_cr.mutable_data<unsigned char>();
    std::memcpy(out_data, (void *)(bands[2].dct), ::dali::volume(dct_cr_shape));
    delete[] bands[2].dct;
  }
  // ::dali::TypeInfo type = input.type();
  // auto &tp = ws.GetThreadPool();
  // const auto &in_shape = input.shape();
  // for (int sample_id = 0; sample_id < in_shape.num_samples(); sample_id++) {
  //   tp.AddWork(
  //       [&, sample_id](int thread_id) {
  //         type.Copy<::dali::CPUBackend, ::dali::CPUBackend>(output.raw_mutable_tensor(sample_id),
  //                                                           input.raw_tensor(sample_id),
  //                                                           in_shape.tensor_size(sample_id), 0);
  //       },
  //       in_shape.tensor_size(sample_id));
  // }
  // tp.RunAll();
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
