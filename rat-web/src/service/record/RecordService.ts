import RecordProvider from "@/service/record/RecordProvider";
import type {GenericResponse} from "@/service/response/GenericResponse";
import type {CameraConfig, HeartRateConfig, HeartRateData, Latency, RecordStatus} from "@/service/record/Data";
import axios from "axios";
import {resolve} from "chart.js/helpers";

export default class RecordService extends RecordProvider{
    constructor(baseUrl: string) {
        super(baseUrl);
    }

    onHeartRateConfigUpdate(onData: (config: HeartRateConfig) => void): void {
        const eventSource = new EventSource(this.baseUrl + "/heart-rate/config-realtime");

        eventSource.onmessage = (event: MessageEvent) => {
            try {
                onData(JSON.parse(event.data) as HeartRateConfig)
                console.log("Config update received:", event.data);
            } catch (error) {
                console.error("Failed to parse config update:", error);
            }
        };
    }

    onHeartRateData(onData: (config: Array<HeartRateData>) => void): void {
        const eventSource = new EventSource(this.baseUrl + "/heart-rate/stream");
        eventSource.onmessage = (event: MessageEvent) => {
            try {
                onData(JSON.parse(event.data) as Array<HeartRateData>)
            } catch (error) {
                console.error("Failed to parse config update:", error);
            }
        };
    }

    updateHeartRateConfig(sampleRateIndex: number, frameSizeIndex: number, frameDelayIndex: number, frameReduceIndex: number): Promise<void> {
        return new Promise((resolve) => {
            axios.post<GenericResponse>(
                this.baseUrl + "/heart-rate/update",
                {
                    sample_rate_index: sampleRateIndex,
                    frame_size_index: frameSizeIndex,
                    frame_delay_index: frameDelayIndex,
                    frame_reduce_index: frameReduceIndex,
                },
                { headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                    }}
            ).then(response => {
                resolve();
            })
        })
    }

    getHeartRateConfig(): Promise<HeartRateConfig> {
        return new Promise(resolve => {
            axios.get<GenericResponse<HeartRateConfig>>(
                this.baseUrl + "/heart-rate/config",
                { headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                    }}
            ).then((response) => {
                resolve(response.data.data as HeartRateConfig)
            })
        })
    }

    checkLatency(): Promise<Latency> {
        return new Promise(resolve => {
            const startTime = Date.now();
            axios.get<GenericResponse<number>>(
                this.baseUrl + "/heart-rate/latency",
                { headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                }}
            ).then((response) => {
                const endTime = Date.now();
                resolve({
                    server: endTime - startTime,
                    heartRate: response.data.data
                })
            })
        })
    }

    getCameraConfig(): Promise<CameraConfig> {
        return new Promise((resolve) => {
            axios.get<GenericResponse<CameraConfig>>(
                this.baseUrl + "/camera/config",
                { headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                }}
            ).then(response => {
                resolve(response.data.data as CameraConfig);
            })
        })
    }

    updateCameraConfig(cameraIndex: number, cameraFrameIndex: number, cameraFrameRateIndex: number, lightIndex: number): Promise<void> {
        return new Promise((resolve) => {
            axios.post<GenericResponse>(
                this.baseUrl + "/camera/update",
                {
                    camera_index: cameraIndex,
                    camera_frame_index: cameraFrameIndex,
                    camera_frame_rate_index: cameraFrameRateIndex,
                    light_index: lightIndex,
                },
                { headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                    }}
            ).then(response => {
                resolve();
            })
        })
    }

    onCameraConfigUpdate(onData: (config: CameraConfig) => void): void {
        const eventSource = new EventSource(this.baseUrl + "/camera/config-realtime");

        eventSource.onmessage = (event: MessageEvent) => {
            try {
                onData(JSON.parse(event.data) as CameraConfig)
                console.log("Config update received:", event.data);
            } catch (error) {
                console.error("Failed to parse config update:", error);
            }
        };
    }

    cameraStream(): string {
        return this.baseUrl + "/camera/stream"
    }

    getRecordStatus(): Promise<RecordStatus> {
        return new Promise(resolve => {
            axios.get<GenericResponse<RecordStatus>>(
                this.baseUrl + "/record/status",
                { headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                    }}
            ).then((response) => {
                resolve(response.data.data as RecordStatus)
            })
        })
    }

    onRecordStatusUpdate(onData: (status: RecordStatus) => void): void {
        const eventSource = new EventSource(this.baseUrl + "/record/status-realtime");

        eventSource.onmessage = (event: MessageEvent) => {
            try {
                onData(JSON.parse(event.data) as RecordStatus)
                console.log("Config update received:", event.data);
            } catch (error) {
                console.error("Failed to parse config update:", error);
            }
        };
    }

    recordStart(duration: number): Promise<void> {
        return new Promise((resolve) => {
            axios.post<GenericResponse>(
                this.baseUrl + "/record/start",
                {
                    duration: duration,
                },
                { headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                    }}
            ).then(response => {
                resolve();
            })
        })
    }

    recordStop(): Promise<void> {
        return new Promise((resolve) => {
            axios.post<GenericResponse>(
                this.baseUrl + "/record/stop",
                {},
                { headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                    }}
            ).then(response => {
                resolve();
            })
        })
    }

    getRecords(): Promise<Array<string>> {
        return new Promise(resolve => {
            axios.get<GenericResponse<Array<string>>>(
                this.baseUrl + "/record/records",
                { headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                    }}
            ).then((response) => {
                resolve(response.data.data as Array<string>)
            })
        })
    }

    onRecordsUpdate(onData: (recordStatus: Array<string>) => void): void {
        const eventSource = new EventSource(this.baseUrl + "/record/records-realtime");

        eventSource.onmessage = (event: MessageEvent) => {
            try {
                onData(JSON.parse(event.data) as Array<string>)
                console.log("Config update received:", event.data);
            } catch (error) {
                console.error("Failed to parse config update:", error);
            }
        };
    }
}
