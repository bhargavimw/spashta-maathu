export interface RemovedRegion {
  start: number;
  end: number;
  reason: string;
}

export interface WordToken {
  i: number;
  text: string;
  start: number;
  end: number;
  gap_after?: number;
}

export interface ProcessResult {
  clean_audio_url: string;
  clean_audio_path: string;
  original_duration: number;
  clean_duration: number;
  time_saved: number;
  removed_regions: RemovedRegion[];
  words: WordToken[] | null;
  verbatim_text: string | null;
  clean_text: string | null;
  diff: unknown | null;
}

export interface HealthStatus {
  status: string;
  stages: {
    ingest: boolean;
    vad: boolean;
    gap_rules: boolean;
    splicer: boolean;
    asr: boolean;
    align: boolean;
    classifier: boolean;
  };
  version: string;
}
