#include "TF_Model.H"


int ModelCreate(model_t* model, const char* graph_def_filename) {
  model->status = TF_NewStatus();
  model->graph = TF_NewGraph();

  {
    // Create the session.
    TF_SessionOptions* opts = TF_NewSessionOptions();
    model->session = TF_NewSession(model->graph, opts, model->status);
    TF_DeleteSessionOptions(opts);
    if (!Okay(model->status)) return 0;
  }

  TF_Graph* g = model->graph;

  {
    // Import the graph.
    TF_Buffer* graph_def = ReadFile(graph_def_filename);
    if (graph_def == NULL) return 0;
    printf("Read GraphDef of %zu bytes\n", graph_def->length);
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(g, graph_def, opts, model->status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(graph_def);
    if (!Okay(model->status)) return 0;
  }

  // Handles to the interesting operations in the graph.
  model->input.oper = TF_GraphOperationByName(g, "input");
  model->input.index = 0;
  model->target.oper = TF_GraphOperationByName(g, "target");
  model->target.index = 0;
  model->output.oper = TF_GraphOperationByName(g, "output");
  model->output.index = 0;

  model->init_op = TF_GraphOperationByName(g, "init");
  model->train_op = TF_GraphOperationByName(g, "train");
  model->save_op = TF_GraphOperationByName(g, "save/control_dependency");
  model->restore_op = TF_GraphOperationByName(g, "save/restore_all");

  model->checkpoint_file.oper = TF_GraphOperationByName(g, "save/Const");
  model->checkpoint_file.index = 0;
  return 1;
}

void ModelDestroy(model_t* model) {
  TF_DeleteSession(model->session, model->status);
  Okay(model->status);
  TF_DeleteGraph(model->graph);
  TF_DeleteStatus(model->status);
}

int ModelInit(model_t* model) {
  const TF_Operation* init_op[1] = {model->init_op};
  TF_SessionRun(model->session, NULL,
                /* No inputs */
                NULL, NULL, 0,
                /* No outputs */
                NULL, NULL, 0,
                /* Just the init operation */
                init_op, 1,
                /* No metadata */
                NULL, model->status);
  return Okay(model->status);
}

int ModelCheckpoint(model_t* model, const char* checkpoint_prefix, int type) {
  TF_Tensor* t = ScalarStringTensor(checkpoint_prefix, model->status);
  if (!Okay(model->status)) {
    TF_DeleteTensor(t);
    return 0;
  }
  TF_Output inputs[1] = {model->checkpoint_file};
  TF_Tensor* input_values[1] = {t};
  const TF_Operation* op[1] = {type == SAVE ? model->save_op
                                            : model->restore_op};
  TF_SessionRun(model->session, NULL, inputs, input_values, 1,
                /* No outputs */
                NULL, NULL, 0,
                /* The operation */
                op, 1, NULL, model->status);
  TF_DeleteTensor(t);
  return Okay(model->status);
}

int ModelPredict(model_t* model, float* batch, int batch_size) {
  // batch consists of 1x1 matrices.
  printf("Batch size %d\n", batch_size);

  const int64_t ip_dims[3] = {batch_size, 1, 9}; // Important - must match python
  const size_t ip_nbytes = batch_size * sizeof(float)*9; //9D input
  const size_t op_nbytes = batch_size * sizeof(float)*1; //1D output

  TF_Tensor* t = TF_AllocateTensor(TF_FLOAT, ip_dims, 3, ip_nbytes);

  memcpy(TF_TensorData(t), batch, ip_nbytes);

  TF_Output inputs[1] = {model->input};
  TF_Tensor* input_values[1] = {t};
  TF_Output outputs[1] = {model->output};
  TF_Tensor* output_values[1] = {NULL};

  TF_SessionRun(model->session, NULL, inputs, input_values, 1, outputs,
                output_values, 1,
                /* No target operations to run */
                NULL, 0, NULL, model->status);
  TF_DeleteTensor(t);
  if (!Okay(model->status)) return 0;

  if (TF_TensorByteSize(output_values[0]) != op_nbytes) {
    fprintf(stderr,
            "ERROR: Expected predictions tensor to have %zu bytes, has %zu\n",
            op_nbytes, TF_TensorByteSize(output_values[0]));
    TF_DeleteTensor(output_values[0]);
    return 0;
  }

  float* predictions = (float*)malloc(op_nbytes);
  memcpy(predictions, TF_TensorData(output_values[0]), op_nbytes);
  TF_DeleteTensor(output_values[0]);

  printf("Predictions:\n");
  for (int i = 0; i < batch_size; ++i) {
    // for (int j = 0; j < 9; ++j)
    // {
    //   printf("\t x1 = %f \n", batch[2*i+j]);
    // }
    printf("\t Cs = %f \n", predictions[i]);
  }
  free(predictions);
  return 1;
}

void NextBatchForTraining(TF_Tensor** inputs_tensor,
                          TF_Tensor** targets_tensor,
                          float* inputs, float* targets) {
#define BATCH_SIZE 100
    
  const int64_t ip_dims[] = {BATCH_SIZE, 1, 9}; // Important
  size_t ip_nbytes = BATCH_SIZE * sizeof(float)*9;
  
  const int64_t op_dims[] = {BATCH_SIZE, 1, 1}; // Important
  size_t op_nbytes = BATCH_SIZE * sizeof(float)*1;
  
  *inputs_tensor = TF_AllocateTensor(TF_FLOAT, ip_dims, 3, ip_nbytes);
  *targets_tensor = TF_AllocateTensor(TF_FLOAT, op_dims, 3, op_nbytes);
  
  memcpy(TF_TensorData(*inputs_tensor), inputs, ip_nbytes);
  memcpy(TF_TensorData(*targets_tensor), targets, op_nbytes);

#undef BATCH_SIZE
}

int ModelRunTrainStep(model_t* model, float* batch, float* targets) {
  TF_Tensor *x, *y;
  NextBatchForTraining(&x, &y, batch, targets);
  TF_Output inputs[2] = {model->input, model->target};
  TF_Tensor* input_values[2] = {x, y};
  const TF_Operation* train_op[1] = {model->train_op};
  TF_SessionRun(model->session, NULL, inputs, input_values, 2,
                /* No outputs */
                NULL, NULL, 0, train_op, 1, NULL, model->status);
  TF_DeleteTensor(x);
  TF_DeleteTensor(y);
  return Okay(model->status);
}

int Okay(TF_Status* status) {
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: %s\n", TF_Message(status));
    return 0;
  }
  return 1;
}

TF_Buffer* ReadFile(const char* filename) {
  int fd = open(filename, 0);
  if (fd < 0) {
    perror("failed to open file: ");
    return NULL;
  }
  struct stat stat;
  if (fstat(fd, &stat) != 0) {
    perror("failed to read file: ");
    return NULL;
  }
  char* data = (char*)malloc(stat.st_size);
  ssize_t nread = read(fd, data, stat.st_size);
  if (nread < 0) {
    perror("failed to read file: ");
    free(data);
    return NULL;
  }
  if (nread != stat.st_size) {
    fprintf(stderr, "read %zd bytes, expected to read %zd\n", nread,
            stat.st_size);
    free(data);
    return NULL;
  }
  TF_Buffer* ret = TF_NewBufferFromString(data, stat.st_size);
  free(data);
  return ret;
}

TF_Tensor* ScalarStringTensor(const char* str, TF_Status* status) {
  size_t nbytes = 8 + TF_StringEncodedSize(strlen(str));
  TF_Tensor* t = TF_AllocateTensor(TF_STRING, NULL, 0, nbytes);
  char* data = (char*)TF_TensorData(t);
  memset(data, 0, 8);  // 8-byte offset of first string.
  TF_StringEncode(str, strlen(str), data + 8, nbytes - 8, status);
  return t;
}

int DirectoryExists(const char* dirname) {
  struct stat buf;
  return stat(dirname, &buf) == 0;
}