from pipeline import RagPipeline, set_shared_pipeline


def preload_models() -> None:
    set_shared_pipeline(RagPipeline())
