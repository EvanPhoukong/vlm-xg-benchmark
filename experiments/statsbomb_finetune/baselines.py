"""Fit statistical controls on training only. Test scoring is intentionally deferred."""
import json

from common import OUT, check_data, fit_logistic, predict, metrics, freeze, save, sha

FEATURES = {"image_feature_logistic":["distance","angle","defenders_in_cone"],
            "logistic_with_header":["distance","angle","defenders_in_cone","header"]}


def main():
    protocol, data = check_data()
    train, validation = data["train"], data["validation"]
    prior = sum(r["goal"] for r in train)/len(train)
    models = {name:fit_logistic(train, features) for name,features in FEATURES.items()}
    freeze(OUT/"baseline_models.json", {"models":models,"training_goal_rate":prior,
        "protocol_sha256":sha(OUT/"protocol.json"),"training_shot_ids":[r["shot_id"] for r in train]})
    predictions = {name:predict(model, validation) for name,model in models.items()}
    predictions["training_goal_rate"] = [prior]*len(validation)
    predictions["statsbomb_xg"] = [r["statsbomb_xg"] for r in validation]
    summary = {name:metrics([r["goal"] for r in validation],p) for name,p in predictions.items()}
    save(OUT/"baseline_validation.json", {"split":"validation","scores":summary,"test_scored":False})
    save(OUT/"baseline_validation_predictions.json", [{"shot_id":r["shot_id"],"goal":r["goal"],
        **{name:p[i] for name,p in predictions.items()}} for i,r in enumerate(validation)])
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
