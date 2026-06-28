"""TAMER fine-tuning module with replay-aware validation and fixed metrics."""

from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
import torch.optim as optim

from tamer.datamodule import Batch, vocab
from tamer.lit_tamer import LitTAMER
from tamer.university.metrics import write_metric_report
from tamer.utils.utils import ExpRateRecorder, ce_loss, to_bi_tgt_out, to_struct_output


class LitUniversityTAMER(LitTAMER):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        learning_rate: float = 5e-5,
        patience: int = 4,
        milestones: List[int] = (10, 16),
        vocab_size: int = 248,
        weight_decay: float = 1e-4,
        freeze_encoder_epochs: int = 2,
        hme_baseline_exprate: float = 0.6954,
        max_hme_drop: float = 0.02,
        retention_penalty: float = 10.0,
        prediction_output_dir: str = "outputs/predictions",
    ) -> None:
        super().__init__(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            beam_size=beam_size,
            max_len=max_len,
            alpha=alpha,
            early_stopping=early_stopping,
            temperature=temperature,
            learning_rate=learning_rate,
            patience=patience,
            milestones=list(milestones),
            vocab_size=vocab_size,
        )
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.hme_baseline_exprate = hme_baseline_exprate
        self.max_hme_drop = max_hme_drop
        self.retention_penalty = retention_penalty
        self.prediction_output_dir = prediction_output_dir
        self.university_exprate = ExpRateRecorder()
        self.hme_exprate = ExpRateRecorder()

    def on_train_epoch_start(self) -> None:
        frozen = self.current_epoch < self.freeze_encoder_epochs
        for parameter in self.tamer_model.encoder.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.tamer_model.encoder.eval()
        self.log("encoder_frozen", float(frozen), prog_bar=False, on_step=False, on_epoch=True)

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        struct_out, _ = to_struct_output(batch.indices, self.device)
        out_hat, sim = self(batch.imgs, batch.mask, tgt)
        loss = ce_loss(out_hat, out) + ce_loss(sim, struct_out, ignore_idx=-1)
        prefix = "university" if dataloader_idx == 0 else "hme"
        self.log("val_{}_loss".format(prefix), loss, on_step=False, on_epoch=True, sync_dist=True)
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        recorder = self.university_exprate if dataloader_idx == 0 else self.hme_exprate
        recorder([hyp.seq for hyp in hyps], batch.indices)
        self.log(
            "val_{}_ExpRate".format(prefix),
            recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def validation_epoch_end(self, outputs) -> None:
        university = self.university_exprate.compute()
        hme = self.hme_exprate.compute()
        threshold = self.hme_baseline_exprate - self.max_hme_drop
        violation = torch.relu(torch.as_tensor(threshold, device=self.device) - hme)
        score = university - self.retention_penalty * violation
        self.log("val_ExpRate", university, prog_bar=False, sync_dist=True)
        self.log("val_retention_score", score, prog_bar=True, sync_dist=True)
        self.log("val_hme_drop", self.hme_baseline_exprate - hme, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Batch, batch_idx: int):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        records = []
        categories = batch.categories or ["unknown"] * len(batch)
        sources = batch.sources or ["unknown"] * len(batch)
        severities = batch.severities or ["unknown"] * len(batch)
        for name, hyp, truth, category, source, severity in zip(
            batch.img_bases, hyps, batch.indices, categories, sources, severities
        ):
            records.append(
                {
                    "sample_id": name,
                    "pred_tokens": vocab.indices2words(hyp.seq),
                    "gt_tokens": vocab.indices2words(truth),
                    "category": category,
                    "source": source,
                    "severity": severity,
                }
            )
        return records

    def test_epoch_end(self, outputs) -> None:
        records = [record for batch_records in outputs for record in batch_records]
        report = write_metric_report(records, self.prediction_output_dir)
        overall = report["overall"]
        for name in ("ExpRate", "ExpRate_le_1", "ExpRate_le_2", "TokenErrorRate", "ValidLaTeX"):
            self.log("test_" + name, float(overall[name]))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(1, self.hparams.patience // 2),
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_retention_score",
                "interval": "epoch",
                "frequency": 1,
            },
        }
