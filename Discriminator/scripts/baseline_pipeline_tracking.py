"""Small W&B adapter shared by baseline pipeline stages."""

import csv
import os
import re
from contextlib import contextmanager
from pathlib import Path

from omegaconf import OmegaConf

try:
    from .plot_bundles import profiled_plot_path
except ImportError:
    from plot_bundles import profiled_plot_path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience only
    load_dotenv = None


class DisabledRun:
    """W&B-compatible no-op used by tests and explicitly disabled pipelines."""

    def __init__(self):
        self.summary = {}
        self.url = None

    def log(self, *_args, **_kwargs):
        pass


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unnamed"


def parsed_csv_value(value):
    if value == "":
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


class PipelineTracker:
    def __init__(self, cfg, pipeline_id):
        settings = cfg.pipeline.wandb
        self.enabled = bool(settings.get("enabled", True))
        self.mode = str(settings.get("mode", "online"))
        self.project = str(settings.get("project", "weather-discriminator-baselines"))
        self.entity = settings.get("entity")
        self.group = str(pipeline_id)
        self.tags = [str(tag) for tag in settings.get("tags", [])]
        self.pipeline_alias = safe_name(pipeline_id)
        run_dir = cfg.pipeline.get("run_dir")
        self.run_dir = None if run_dir is None else Path(str(run_dir))
        self.logged_artifacts = []
        self._wandb = None
        if self.enabled:
            if load_dotenv is not None:
                load_dotenv(Path(__file__).resolve().parents[2] / "wandb_info.env")
            import wandb
            self._wandb = wandb
            if self.mode == "online" and not wandb.login():
                raise RuntimeError("W&B online mode is enabled, but authentication failed.")

    @contextmanager
    def run(self, name, job_type, cfg, metadata=None, tags=None):
        if not self.enabled:
            yield DisabledRun()
            return
        config = OmegaConf.to_container(cfg, resolve=True)
        if metadata:
            config = {"configuration": config, "stage": dict(metadata)}
        run = self._wandb.init(
            project=self.project,
            entity=None if self.entity is None else str(self.entity),
            group=self.group,
            job_type=str(job_type),
            name=f"{self.pipeline_alias}/{name}",
            tags=self.tags + [f"pipeline:{self.pipeline_alias}"] + [str(tag) for tag in (tags or [])],
            config=config,
            mode=self.mode,
            save_code=True,
            reinit="create_new",
            **({"dir": str(self.run_dir / "wandb")} if self.run_dir is not None else {}),
        )
        try:
            run.summary["pipeline_id"] = self.group
            run.summary["pipeline_run_directory"] = self.pipeline_alias
            run.summary["status"] = "running"
            yield run
        except BaseException as error:
            run.summary["status"] = "failed"
            run.summary["error"] = f"{type(error).__name__}: {error}"
            run.finish(exit_code=1)
            raise
        else:
            run.summary["status"] = "completed"
            run.finish()

    def log_artifact(self, run, name, artifact_type, paths, metadata=None):
        paths = [Path(path) for path in paths if Path(path).is_file()]
        if not self.enabled or not paths:
            return None
        artifact_metadata = {"pipeline_id": self.group}
        artifact_metadata.update(dict(metadata or {}))
        artifact = self._wandb.Artifact(
            f"{self.pipeline_alias}-{safe_name(name)}", type=str(artifact_type), metadata=artifact_metadata,
        )
        common_root = Path(Path(paths[0]).anchor)
        try:
            common_root = Path(os.path.commonpath([str(path.parent) for path in paths]))
        except ValueError:
            pass
        for path in paths:
            try:
                artifact_name = str(path.relative_to(common_root))
            except ValueError:
                artifact_name = path.name
            artifact.add_file(str(path), name=artifact_name)
        logged = run.log_artifact(
            artifact, aliases=[self.pipeline_alias]
        )
        self.logged_artifacts.append(logged)
        return logged

    def log_csv_table(self, run, key, path):
        path = Path(path)
        if not self.enabled or not path.is_file():
            return
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if rows:
            data = [[parsed_csv_value(value) for value in row] for row in rows[1:]]
            run.log({str(key): self._wandb.Table(columns=rows[0], data=data)})

    def log_images(self, run, paths, root):
        if not self.enabled:
            return
        root = Path(root)
        for path in paths:
            path = Path(path)
            if not path.is_file():
                path = profiled_plot_path(path)
            if not path.is_file():
                continue
            key = "plots/" + str(path.relative_to(root).with_suffix(""))
            run.log({key: self._wandb.Image(str(path))})

    def log_records_table(self, run, key, records):
        if not self.enabled or not records:
            return
        columns = sorted({column for record in records for column in record})
        # wandb.Table infers one type per column and rejects rows that disagree.
        # Interpretability rows mix types within a column -- lead_hour is an int for
        # forecast cases but None/"" for cases with no lead context, and severity is
        # a float for corruptions but "" for forecasts -- which trips a "Number not
        # assignable to None or String" error. Coerce any column that carries a
        # string to strings throughout (keeping None as None); purely numeric or
        # all-None columns are left untouched.
        stringify = {
            column for column in columns
            if any(isinstance(record.get(column), str) for record in records)
        }
        def cell(column, value):
            if value is None or column not in stringify:
                return value
            return str(value)
        data = [[cell(column, record.get(column)) for column in columns] for record in records]
        run.log({str(key): self._wandb.Table(columns=columns, data=data)})
