#include "image_classifier.h"
#include "tensorflow/cc/framework/gradients.h"

#define DT tf::DataType

ImageClassifier::ImageClassifier(const int &batch_size, const int &image_width, const int &image_height,
                                 const int &image_chenal, const int &nof_class)
    : m_batch_size(batch_size), m_image_width(image_width), m_image_height(image_height),
      m_image_chenal(image_chenal), m_nof_class(nof_class)
{
    this->build_fc_model();
    // this->build_cnn_model();
}

ImageClassifier::~ImageClassifier()
{
    delete this->m_p_plcaeholder_filenames;
    delete this->m_p_placeholder_labels;
    delete this->m_p_session;
}

void ImageClassifier::train(const std::vector<std::string> &filenames, const std::vector<int> &labels,
                            float &accuracy, float &loss)
{
    // train
    std::vector<tf::Tensor> outputs;
    this->run(filenames, labels, this->m_outputlist, outputs);
    accuracy = *outputs[0].scalar<float>().data();
    loss = *outputs[1].scalar<float>().data();
}

void ImageClassifier::test(const std::vector<std::string> &filenames, const std::vector<int> &labels, float &accuracy)
{
    // test
    std::vector<tf::Tensor> outputs;
    this->run(filenames, labels, {this->m_outputlist[0]}, outputs);
    accuracy = *outputs[0].scalar<float>().data();
}

void ImageClassifier::build_fc_model()
{
    // input layer
    auto root = tf::Scope::NewRootScope();
    this->m_p_plcaeholder_filenames = new tfop::Placeholder(root, DT::DT_STRING);
    this->m_p_placeholder_labels = new tfop::Placeholder(root, DT::DT_INT32);
    auto list_filename = tfop::Unstack(root, *this->m_p_plcaeholder_filenames, this->m_batch_size);
    auto image_size = this->m_image_width * this->m_image_height * this->m_image_chenal;
    std::vector<tf::Output> vct_image;
    for (auto i = 0; i < this->m_batch_size; i++)
    {
        auto file_reader = tfop::ReadFile(root, list_filename[i]);
        auto image_reader = tfop::DecodePng(root, file_reader,
                                            tfop::DecodePng::Channels(this->m_image_chenal));
        auto reshaped_image = tfop::Reshape(root, image_reader, {image_size});
        auto normalized_image = tfop::Div(root, tfop::Cast(root, reshaped_image, DT::DT_FLOAT), 255.f);
        vct_image.push_back(normalized_image);
    }
    tf::Output data_flow = tfop::Stack(root, vct_image);
    auto labels = tfop::OneHot(root, *this->m_p_placeholder_labels, this->m_nof_class, 1.f, 0.f);

    // fully connection layer
    auto w_fc = tfop::Variable(root, {image_size, this->m_nof_class}, DT::DT_FLOAT);
    auto init_w_fc = tfop::RandomNormal(root, {image_size, this->m_nof_class}, DT::DT_FLOAT);
    auto assign_w_fc = tfop::Assign(root, w_fc, init_w_fc);
    auto b_fc = tfop::Variable(root, {this->m_nof_class}, DT::DT_FLOAT);
    auto init_b_fc = tfop::Const(root, 0.f, {this->m_nof_class});
    auto assign_b_fc = tfop::Assign(root, b_fc, init_b_fc);
    data_flow = tfop::MatMul(root, data_flow, w_fc);
    data_flow = tfop::Add(root, data_flow, b_fc);
    data_flow = tfop::Relu(root, data_flow);
    auto logits = data_flow;

    // accuracy
    auto result = tfop::Equal(root, tfop::ArgMax(root, labels, 1), tfop::ArgMax(root, logits, 1));
    this->m_outputlist.push_back(tfop::ReduceMean(root, tfop::Cast(root, result, DT::DT_FLOAT), 0));

    // loss function
    auto softmax = tfop::Softmax(root, logits);
    auto loss = tfop::Square(root, tfop::Sub(root, softmax, labels));
    this->m_outputlist.push_back(tfop::ReduceSum(root, loss, {0, 1}));

    // gradients
    std::vector<tf::Output> grad_outputs;
    TF_CHECK_OK(tf::AddSymbolicGradients(root, {loss}, {w_fc, b_fc}, &grad_outputs));
    auto learn_rate = tfop::Const(root, 0.01f, {});
    this->m_outputlist.push_back(tfop::ApplyGradientDescent(root, w_fc, learn_rate, grad_outputs[0]));
    this->m_outputlist.push_back(tfop::ApplyGradientDescent(root, b_fc, learn_rate, grad_outputs[1]));

    // session
    this->m_p_session = new tf::ClientSession(root);
    TF_CHECK_OK(this->m_p_session->Run({assign_w_fc, assign_b_fc}, nullptr));
}

void ImageClassifier::build_cnn_model()
{
    // input layer
    auto root = tf::Scope::NewRootScope();
    this->m_p_plcaeholder_filenames = new tfop::Placeholder(root, DT::DT_STRING);
    this->m_p_placeholder_labels = new tfop::Placeholder(root, DT::DT_INT32);
    auto list_filename = tfop::Unstack(root, *this->m_p_plcaeholder_filenames, this->m_batch_size);
    std::vector<tf::Output> vct_image;
    for (auto i = 0; i < this->m_batch_size; i++)
    {
        auto file_reader = tfop::ReadFile(root, list_filename[i]);
        auto image_reader = tfop::DecodePng(root, file_reader,
                                            tfop::DecodePng::Channels(this->m_image_chenal));
        auto reshaped_image = tfop::Reshape(root, image_reader,
                                            {this->m_image_width, this->m_image_height, this->m_image_chenal});
        auto normalized_image = tfop::Div(root, tfop::Cast(root, reshaped_image, DT::DT_FLOAT), 255.f);
        vct_image.push_back(normalized_image);
    }
    // data_flow ==> [64*28*28*1]
    tf::Output data_flow = tfop::Stack(root, vct_image);
    auto labels = tfop::OneHot(root, *this->m_p_placeholder_labels, this->m_nof_class, 1.f, 0.f);

    // conv layer 1 < filter=[5*5*1*32] strides=[1*1*1*1] ==> [64*28*28*32]
    data_flow = tfop::Conv2D(root, data_flow, tf::Tensor(DT::DT_FLOAT, {5, 5, 1, 32}), {1, 1, 1, 1}, "SAME");
    auto b_conv1 = tfop::Variable(root, {32}, DT::DT_FLOAT);
    auto init_b_conv1 = tfop::Const(root, 0.f, {32});
    auto assign_b_conv1 = tfop::Assign(root, b_conv1, init_b_conv1);
    data_flow = tfop::Relu(root, tfop::Add(root, data_flow, b_conv1));

    // max pool 1 < ksize=[1*2*2*1] strides=[1*2*2*1] ==> [64*14*14*32]
    data_flow = tfop::MaxPool(root, data_flow, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME");

    // conv layer 2 < filter=[5*5*32*64] strides=[1*1*1*1] ==> [64*14*14*64]
    data_flow = tfop::Conv2D(root, data_flow, tf::Tensor(DT::DT_FLOAT, {5, 5, 32, 64}), {1, 1, 1, 1}, "SAME");
    auto b_conv2 = tfop::Variable(root, {64}, DT::DT_FLOAT);
    auto init_b_conv2 = tfop::Const(root, 0.f, {64});
    auto assign_b_conv2 = tfop::Assign(root, b_conv2, init_b_conv2);
    data_flow = tfop::Relu(root, tfop::Add(root, data_flow, b_conv2));

    // max pool 2 < ksize=[1*2*2*1] strides=[1*2*2*1] ==> [64*7*7*64]
    data_flow = tfop::MaxPool(root, data_flow, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME");

    // flat ==> [64, 3136]
    auto size = this->m_image_width * this->m_image_height * 64 / 4 / 4;
    data_flow = tfop::Reshape(root, data_flow, {64, size});

    // fully connection layer 1 ==> [64, 256]
    auto w_fc1 = tfop::Variable(root, {size, 256}, DT::DT_FLOAT);
    auto init_w_fc1 = tfop::RandomNormal(root, {size, 256}, DT::DT_FLOAT);
    auto assign_w_fc1 = tfop::Assign(root, w_fc1, init_w_fc1);
    auto b_fc1 = tfop::Variable(root, {256}, DT::DT_FLOAT);
    auto init_b_fc1 = tfop::Const(root, 0.f, {256});
    auto assign_b_fc1 = tfop::Assign(root, b_fc1, init_b_fc1);
    data_flow = tfop::Add(root, tfop::MatMul(root, data_flow, w_fc1), b_fc1);
    data_flow = tfop::Relu(root, data_flow);

    // fully connection layer 2 ==> [64, 10]
    auto w_fc2 = tfop::Variable(root, {256, 10}, DT::DT_FLOAT);
    auto init_w_fc2 = tfop::RandomNormal(root, {256, 10}, DT::DT_FLOAT);
    auto assign_w_fc2 = tfop::Assign(root, w_fc2, init_w_fc2);
    auto b_fc2 = tfop::Variable(root, {10}, DT::DT_FLOAT);
    auto init_b_fc2 = tfop::Const(root, 0.f, {10});
    auto assign_b_fc2 = tfop::Assign(root, b_fc2, init_b_fc2);
    data_flow = tfop::Add(root, tfop::MatMul(root, data_flow, w_fc2), b_fc2);
    auto logits = data_flow;

    // accuracy ==> []
    auto result = tfop::Equal(root, tfop::ArgMax(root, labels, 1), tfop::ArgMax(root, logits, 1));
    auto accuracy = tfop::ReduceMean(root, tfop::Cast(root, result, DT::DT_FLOAT), 0);
    this->m_outputlist.push_back(accuracy);

    // loss function ==> []
    auto softmax = tfop::Softmax(root, logits);
    auto loss = tfop::ReduceSum(root, tfop::Square(root, tfop::Sub(root, softmax, labels)), {0, 1});
    this->m_outputlist.push_back(loss);

    // gradients
    std::vector<tf::Output> grad_outputs;
    TF_CHECK_OK(tf::AddSymbolicGradients(root, {loss}, {b_conv1, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2}, &grad_outputs));
    auto learn_rate = tfop::Const(root, 0.01f, {});
    this->m_outputlist.push_back(tfop::ApplyGradientDescent(root, b_conv1, learn_rate, grad_outputs[0]));
    this->m_outputlist.push_back(tfop::ApplyGradientDescent(root, b_conv2, learn_rate, grad_outputs[1]));
    this->m_outputlist.push_back(tfop::ApplyGradientDescent(root, w_fc1, learn_rate, grad_outputs[2]));
    this->m_outputlist.push_back(tfop::ApplyGradientDescent(root, b_fc1, learn_rate, grad_outputs[3]));
    this->m_outputlist.push_back(tfop::ApplyGradientDescent(root, w_fc2, learn_rate, grad_outputs[4]));
    this->m_outputlist.push_back(tfop::ApplyGradientDescent(root, b_fc2, learn_rate, grad_outputs[5]));

    // session
    this->m_p_session = new tf::ClientSession(root);
    TF_CHECK_OK(this->m_p_session->Run({assign_b_conv1, assign_b_conv2, assign_w_fc1, assign_b_fc1,
                                        assign_w_fc2, assign_b_fc2},
                                       nullptr));
}

template <typename T>
tf::Tensor ImageClassifier::parse_input(const std::vector<T> &input, const DT &dt)
{
    tf::Tensor tensor(dt, {this->m_batch_size});
    std::copy_n(input.begin(), input.size(), tensor.flat<T>().data());
    return tensor;
}

void ImageClassifier::run(const std::vector<std::string> &filenames, const std::vector<int> &labels,
                          const std::vector<tf::Output> &ins, std::vector<tf::Tensor> &outs)
{
    // parse input
    tf::Tensor input_filename = this->parse_input<std::string>(filenames, DT::DT_STRING);
    tf::Tensor input_label = this->parse_input<int>(labels, DT::DT_INT32);

    // run
    TF_CHECK_OK(this->m_p_session->Run({{*this->m_p_plcaeholder_filenames, input_filename},
                                        {*this->m_p_placeholder_labels, input_label}},
                                       ins, &outs));
}