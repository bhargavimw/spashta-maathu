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
  // Stage 3 extended word fields
  spike_start_s?: number;
  spike_end_s?: number;
  start_s?: number;
  end_s?: number;
  confidence?: number;
  frames?: number;
  suspect?: boolean;
}

export interface TagAnnotation {
  file: string;
  tag: string;
  start_s: number;
  end_s: number;
  text: string;
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
  tags: TagAnnotation[] | null;
}

export interface HealthStatus {
  status: string;
  version: string;
  components: {
    vad: boolean;
    asr: boolean;
    classifier: boolean;
  };
  stages: {
    ingest: boolean;
    vad: boolean;
    gap_rules: boolean;
    splicer: boolean;
    asr: boolean;
    align: boolean;
    classifier: boolean;
    tagger: boolean;
    clean_text: boolean;
    stitch: boolean;
  };
}
