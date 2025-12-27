from typing import Callable, List

from .loops import train_one_epoch_classifier, evaluate_classifier, train_one_epoch_imdb, evaluate_imdb

def quick_tune_classifier(build_model_fn: Callable, make_opt_fn: Callable, train_loader, test_loader,
                          cfgs: List[dict], name: str, device, dtype,
                          tune_batches=150, eval_batches=50) -> dict:
    best = None
    best_metric = float("inf")
    for cfg in cfgs:
        model = build_model_fn().to(device, dtype=dtype)
        opt = make_opt_fn(model, cfg)

        train_one_epoch_classifier(model, train_loader, opt, device=device, dtype=dtype, max_batches=tune_batches)
        te = evaluate_classifier(model, test_loader, device=device, dtype=dtype, max_batches=eval_batches)
        metric = te["loss"]
        print(f"[TUNE] {name} cfg={cfg} val_loss={te['loss']:.4f} val_acc={te['acc']:.4f}")

        if metric < best_metric:
            best_metric = metric
            best = cfg
    print(f"--- [{name}] best cfg = {best} ---")
    return best

def quick_tune_imdb(build_model_fn: Callable, make_opt_fn: Callable, train_loader, test_loader,
                    cfgs: List[dict], name: str, device, dtype,
                    tune_batches=150, eval_batches=50) -> dict:
    best = None
    best_metric = float("inf")
    for cfg in cfgs:
        model = build_model_fn().to(device, dtype=dtype)
        opt = make_opt_fn(model, cfg)

        train_one_epoch_imdb(model, train_loader, opt, device=device, dtype=dtype, max_batches=tune_batches)
        te = evaluate_imdb(model, test_loader, device=device, dtype=dtype, max_batches=eval_batches)
        metric = te["loss"]
        print(f"[TUNE] {name} cfg={cfg} val_loss={te['loss']:.4f} val_acc={te['acc']:.4f}")

        if metric < best_metric:
            best_metric = metric
            best = cfg
    print(f"--- [{name}] best cfg = {best} ---")
    return best
