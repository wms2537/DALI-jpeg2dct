#ifndef EXAMPLE_Jpeg2Dct_H_
#define EXAMPLE_Jpeg2Dct_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/data/types.h"
#include "dali/image/image.h"

namespace jpeg2dct {

class JpegToDct : public ::dali::Operator<::dali::CPUBackend> {
 public:
  inline explicit JpegToDct(const ::dali::OpSpec &spec) :
    ::dali::Operator<::dali::CPUBackend>(spec),
    normalize_(spec.GetArgument<bool>("normalize")),
    channels_(spec.GetArgument<int>("channels")) {}

  virtual inline ~JpegToDct() = default;

  JpegToDct(const JpegToDct&) = delete;
  JpegToDct& operator=(const JpegToDct&) = delete;
  JpegToDct(JpegToDct&&) = delete;
  JpegToDct& operator=(JpegToDct&&) = delete;

 protected:
  bool CanInferOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::HostWorkspace &ws) override {
    return false;
  }

  void RunImpl(::dali::SampleWorkspace &ws) override;

  bool normalize_;
  int channels_;
};

}  // namespace jpeg2dct

#endif  // EXAMPLE_Jpeg2Dct_H_
