#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"

namespace tf = tensorflow;
namespace tfop = tf::ops;

class ImageClassifier
{
public:
    ImageClassifier(const int &batch_size, const int &image_width, const int &image_height, 
                    const int &image_chenal, const int &nof_class);
    ~ImageClassifier();
    void train(const std::vector<std::string> &filenames, const std::vector<int> &labels, float &accuracy, float &loss);
    void test(const std::vector<std::string> &filenames, const std::vector<int> &labels, float &accuracy);

private:
    void build_fc_model();
    void build_cnn_model();
    template<typename T>
    tf::Tensor parse_input(const std::vector<T> &input, const tf::DataType &dt);

private:
    int m_batch_size;
    int m_image_width;
    int m_image_height;
    int m_image_chenal;
    int m_nof_class;

    tfop::Placeholder *m_p_plcaeholder_filenames;
    tfop::Placeholder *m_p_placeholder_labels;
    std::vector<tf::Output> m_outputlist;
    tf::ClientSession *m_p_session;
};