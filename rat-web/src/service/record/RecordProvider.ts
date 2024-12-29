import type {GenericResponse} from "@/service/response/GenericResponse";
import type {CameraConfig, HeartRateConfig, HeartRateData, Latency, RecordStatus} from "@/service/record/Data";

export default abstract class RecordProvider {
    baseUrl: string
    protected constructor(baseUrl: string) {
        this.baseUrl = baseUrl
    }

    abstract getHeartRateConfig(): Promise<HeartRateConfig>
    abstract updateHeartRateConfig(sampleRateIndex: number, frameSizeIndex: number, frameDelayIndex: number, frameReduceIndex: number): Promise<void>
    abstract onHeartRateData(onData: (data: Array<HeartRateData>) => void): void
    abstract onHeartRateConfigUpdate(onData: (config: HeartRateConfig) => void): void

    abstract getCameraConfig(): Promise<CameraConfig>
    abstract updateCameraConfig(cameraIndex: number, cameraFrameIndex: number, cameraFrameRateIndex: number, lightIndex: number): Promise<void>
    abstract onCameraConfigUpdate(onData: (config: CameraConfig) => void): void
    abstract cameraStream: string

    abstract checkLatency(): Promise<Latency>

    abstract getRecordStatus(): Promise<RecordStatus>
    abstract onRecordStatusUpdate(onData: (status: RecordStatus) => void): void
    abstract recordStart(duration: number): Promise<void>
    abstract recordStop(): Promise<void>

    abstract getRecords(): Promise<Array<string>>
    abstract onRecordsUpdate(onData: (recordStatus: Array<string>) => void): void
}
