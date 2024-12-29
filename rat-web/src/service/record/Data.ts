
export type CameraConfig = {
    CAMERAS: string
    FRAMES: Array<{
        width: number
        height: number
        rate: Array<number>
    }>
    LIGHTS: Array<string>
    camera_index: number
    camera_frame_index: number
    camera_frame_rate_index: number
    light_index: number
}

export type Latency = {
    server: number
    heartRate: number
}

export type HeartRateConfig = {
    SAMPLE_RATES: Array<number>
    FRAME_SIZES: Array<number>
    FRAME_DELAYS: Array<number>
    FRAME_REDUCES: Array<number>
    sample_rate_index: number
    frame_size_index: number
    frame_delay_index: number
    frame_reduce_index: number
}

export type HeartRateData = {
    T: number
    t: number
    p: number
}

export type RecordStatus = {
    is_recording: boolean,
    recording_title: string,
    start_at: string
}
