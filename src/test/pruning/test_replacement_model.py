
import transformers

from neuroflex.pruning.replacement_model import SharedAverageLayerReplacingModelWrapper

if __name__ == "__main__":
    model = transformers.AutoModel.from_pretrained("bert-base-uncased")
    print(model)
    print(model.encoder.layer[0].attention.self.query.weight)
    print(model.encoder.layer[1].attention.self.query.weight)
    print(model.encoder.layer[0].attention.self.query.bias)
    print(model.encoder.layer[1].attention.self.query.bias)

    model_wrapper = SharedAverageLayerReplacingModelWrapper(
        model,
        {
            ("0", "query"): None,
            ("1", "query"): None,
            ("2", "query"): None,
            ("3", "query"): None,
            ("4", "query"): None,
            ("5", "query"): None,
            ("6", "query"): None,
            ("7", "query"): None,
            ("8", "query"): None,
            ("9", "query"): None,
            ("10", "query"): None,
            ("11", "query"): None,
        }
    )
    model_wrapper.replace_layers()
    print((model.encoder.layer[0].attention.self.query.weight +
           model.encoder.layer[1].attention.self.query.weight +
           model.encoder.layer[2].attention.self.query.weight +
           model.encoder.layer[3].attention.self.query.weight +
           model.encoder.layer[4].attention.self.query.weight +
           model.encoder.layer[5].attention.self.query.weight +
           model.encoder.layer[6].attention.self.query.weight +
           model.encoder.layer[7].attention.self.query.weight +
           model.encoder.layer[8].attention.self.query.weight +
           model.encoder.layer[9].attention.self.query.weight +
           model.encoder.layer[10].attention.self.query.weight +
           model.encoder.layer[11].attention.self.query.weight) / 12)
    print(model_wrapper.model.encoder.layer[0].attention.self.query.weight)
    print(model_wrapper.model.encoder.layer[1].attention.self.query.weight)
    print((model.encoder.layer[0].attention.self.query.bias +
           model.encoder.layer[1].attention.self.query.bias +
           model.encoder.layer[2].attention.self.query.bias +
           model.encoder.layer[3].attention.self.query.bias +
           model.encoder.layer[4].attention.self.query.bias +
           model.encoder.layer[5].attention.self.query.bias +
           model.encoder.layer[6].attention.self.query.bias +
           model.encoder.layer[7].attention.self.query.bias +
           model.encoder.layer[8].attention.self.query.bias +
           model.encoder.layer[9].attention.self.query.bias +
           model.encoder.layer[10].attention.self.query.bias +
           model.encoder.layer[11].attention.self.query.bias) / 12)
    print(model_wrapper.model.encoder.layer[0].attention.self.query.bias)
    print(model_wrapper.model.encoder.layer[1].attention.self.query.bias)