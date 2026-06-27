"""Fine-tune the version_3 fusion checkpoint on university mathematics."""

import argparse
from pathlib import Path

import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from tamer.datamodule.university_datamodule import UniversityDataModule
from tamer.lit_university import LitUniversityTAMER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/university_baseline.yaml")
    parser.add_argument("--resume", default=None, help="Resume an interrupted fine-tune checkpoint")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seed_everything(config.get("seed", 7), workers=True)
    data = UniversityDataModule(**config["data"])
    checkpoint = config["pretrained_checkpoint"]
    if not Path(checkpoint).is_file():
        raise FileNotFoundError("version_3 checkpoint not found: {}".format(checkpoint))
    model = LitUniversityTAMER.load_from_checkpoint(
        checkpoint,
        strict=True,
        **config.get("model_overrides", {}),
    )
    output_dir = config.get("output_dir", "outputs/university_baseline")
    trainer_config = dict(config["trainer"])
    if args.resume:
        trainer_config["resume_from_checkpoint"] = args.resume
    early_stopping_patience = trainer_config.pop("early_stopping_patience", 4)
    if config.get("auto_calibrate_hme_baseline", True) and not args.resume:
        calibration_trainer = Trainer(
            logger=False,
            checkpoint_callback=False,
            gpus=trainer_config.get("gpus", 1),
            precision=trainer_config.get("precision", 32),
            deterministic=trainer_config.get("deterministic", True),
            num_sanity_val_steps=0,
        )
        validation_results = calibration_trainer.validate(model, datamodule=data, verbose=False)
        baseline = None
        for result in validation_results:
            for key, value in result.items():
                if key.startswith("val_hme_ExpRate"):
                    baseline = float(value)
        if baseline is None:
            raise RuntimeError("Could not calibrate version_3 on the fixed HME validation cache")
        model.hme_baseline_exprate = baseline
        model.hparams.hme_baseline_exprate = baseline
        print("Calibrated version_3 HME validation ExpRate: {:.6f}".format(baseline))
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(output_dir) / "checkpoints"),
        filename="{epoch:02d}-{val_retention_score:.4f}",
        monitor="val_retention_score",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val_retention_score",
            mode="max",
            patience=early_stopping_patience,
        ),
    ]
    trainer = Trainer(
        callbacks=callbacks,
        logger=CSVLogger(save_dir=output_dir, name="logs"),
        **trainer_config,
    )
    trainer.fit(model, datamodule=data)
    print("Best retention-aware checkpoint: {}".format(checkpoint_callback.best_model_path))


if __name__ == "__main__":
    main()
