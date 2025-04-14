import type { GenericResponse } from "@/service/response/GenericResponse";
import type { CameraConfig, HeartRateConfig, HeartRateData, Latency, RecordStatus } from "@/service/record/Data";
import axios from "axios";
import type { AxiosResponse } from "axios"; // Type-only import

export interface RecordSummary { id: string; }
export interface InterpolatedData { 
    time: number[]; 
    value: (number | null)[]; 
    normalized?: (number | null)[];
    mean?: number;
    std?: number;
    dominant_freq_hz?: number;
}
export interface Annotation { time: number; value: number; type: 'peak' | 'valley'; }
export interface AnnotationsResponse { annotations: Annotation[]; }
export interface RecordConfig { duration?: number; 'heart-rate-sensor'?: { sample_rate?: number }; }

interface ApiResponse<T> {
    code: string;
    message: string;
    data?: T;
}

export default class RecordService {
    protected baseUrl: string;
    private recordsUpdateCallback: ((records: string[]) => void) | null = null;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
    }

    onHeartRateConfigUpdate(onData: (config: HeartRateConfig) => void): void {
        const eventSource = new EventSource(this.baseUrl + "/heart-rate/config-realtime");
        eventSource.onmessage = (event: MessageEvent) => {
            try {
                onData(JSON.parse(event.data) as HeartRateConfig)
            } catch (error) {
                console.error("Failed to parse heart rate config update:", error);
            }
        };
        eventSource.onerror = (error) => {
            console.error("Heart Rate Config SSE error:", error);
            eventSource.close();
        };
    }

    onHeartRateData(onData: (data: Array<HeartRateData>) => void): void {
        const eventSource = new EventSource(this.baseUrl + "/heart-rate/stream");
        eventSource.onmessage = (event: MessageEvent) => {
            try {
                onData(JSON.parse(event.data) as Array<HeartRateData>)
            } catch (error) {
                console.error("Failed to parse heart rate data update:", error);
            }
        };
        eventSource.onerror = (error) => {
            console.error("Heart Rate Data SSE error:", error);
            eventSource.close();
        };
    }

    updateHeartRateConfig(sampleRateIndex: number, frameSizeIndex: number, frameDelayIndex: number, frameReduceIndex: number): Promise<void> {
        return axios.post<GenericResponse<any>>(
            this.baseUrl + "/heart-rate/update",
            {
                sample_rate_index: sampleRateIndex,
                frame_size_index: frameSizeIndex,
                frame_delay_index: frameDelayIndex,
                frame_reduce_index: frameReduceIndex,
            },
            { headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' } }
        ).then(() => Promise.resolve())
         .catch(error => {
             console.error("Error updating heart rate config:", error);
             return Promise.reject(error);
         });
    }

    getHeartRateConfig(): Promise<HeartRateConfig> {
        return axios.get<HeartRateConfig>(
            this.baseUrl + "/heart-rate/config",
            { headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' } }
        ).then(response => response.data)
         .catch(error => {
             console.error("Error fetching heart rate config:", error);
             return Promise.reject(error);
         });
    }

    checkLatency(): Promise<Latency> {
        return new Promise((resolve, reject) => {
            const startTime = Date.now();
            axios.get<number>(
                this.baseUrl + "/heart-rate/latency",
                { headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' } }
            ).then((response) => {
                const endTime = Date.now();
                const latencyData = response.data;
                if (typeof latencyData === 'number') {
                    resolve({
                        server: endTime - startTime,
                        heartRate: latencyData
                    });
                } else {
                    reject(new Error('Invalid latency data received'));
                }
            }).catch(error => {
                console.error("Error fetching latency:", error);
                reject(error);
            });
        });
    }

    getCameraConfig(): Promise<CameraConfig> {
        return axios.get<CameraConfig>(
            this.baseUrl + "/camera/config",
            { headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' } }
        ).then(response => response.data)
         .catch(error => {
             console.error("Error fetching camera config:", error);
             return Promise.reject(error);
         });
    }

    updateCameraConfig(cameraIndex: number, cameraFrameIndex: number, cameraFrameRateIndex: number, lightIndex: number): Promise<void> {
        return axios.post<GenericResponse<any>>(
            this.baseUrl + "/camera/update",
            {
                camera_index: cameraIndex,
                camera_frame_index: cameraFrameIndex,
                camera_frame_rate_index: cameraFrameRateIndex,
                light_index: lightIndex,
            },
            { headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' } }
        ).then(() => Promise.resolve())
         .catch(error => {
             console.error("Error updating camera config:", error);
             return Promise.reject(error);
         });
    }

    onCameraConfigUpdate(onData: (config: CameraConfig) => void): void {
        const eventSource = new EventSource(this.baseUrl + "/camera/config-realtime");
        eventSource.onmessage = (event: MessageEvent) => {
            try {
                onData(JSON.parse(event.data) as CameraConfig)
            } catch (error) {
                console.error("Failed to parse camera config update:", error);
            }
        };
        eventSource.onerror = (error) => {
            console.error("Camera Config SSE error:", error);
            eventSource.close();
        };
    }

    get cameraStream(): string {
        return this.baseUrl + "/camera/stream";
    }

    getRecordStatus(): Promise<RecordStatus> {
        return axios.get<RecordStatus>(
            this.baseUrl + "/record/status",
            { headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' } }
        ).then(response => response.data)
         .catch(error => {
             console.error("Error fetching record status:", error);
             return Promise.reject(error);
         });
    }

    onRecordStatusUpdate(onData: (status: RecordStatus) => void): void {
        const eventSource = new EventSource(this.baseUrl + "/record/status-realtime");
        eventSource.onmessage = (event: MessageEvent) => {
            try {
                onData(JSON.parse(event.data) as RecordStatus)
            } catch (error) {
                console.error("Failed to parse record status update:", error);
            }
        };
        eventSource.onerror = (error) => {
            console.error("Record Status SSE error:", error);
            eventSource.close();
        };
    }

    recordStart(duration: number): Promise<void> {
        return axios.post<GenericResponse<any>>(
            this.baseUrl + "/record/start",
            { duration: duration },
            { headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' } }
        ).then(() => Promise.resolve())
         .catch(error => {
             console.error("Error starting recording:", error);
             return Promise.reject(error);
         });
    }

    recordStop(): Promise<void> {
        return axios.post<GenericResponse<any>>(
            this.baseUrl + "/record/stop",
            {},
            { headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' } }
        ).then(() => Promise.resolve())
         .catch(error => {
             console.error("Error stopping recording:", error);
             return Promise.reject(error);
         });
    }

    async getRecords(): Promise<string[]> {
        try {
            const response = await axios.get<ApiResponse<string[]>>(`${this.baseUrl}/records`);
            if (response.data?.code === 'OK' && Array.isArray(response.data.data)) {
                 return response.data.data;
            } else {
                console.warn('Failed to get records or unexpected format:', response.data?.message);
                return [];
            }
        } catch (error) {
            console.error("Failed to fetch records:", error);
            return [];
        }
    }

    onRecordsUpdate(callback: (records: string[]) => void): void {
        this.recordsUpdateCallback = callback;
        console.warn('Real-time record updates not fully implemented in RecordService.');
    }

    async getRecordConfig(recordId: string): Promise<RecordConfig | null> {
        try {
            const path = `${this.baseUrl}/records/${recordId}`;
            console.log(`Attempting to fetch record config from: ${path}`);
            
            const response = await axios.get<ApiResponse<RecordConfig>>(path);
            if (response.data?.code === 'OK' && response.data.data) {
                return response.data.data;
            } else {
                console.warn(`Failed to get config for ${recordId}:`, response.data?.message);
                return null;
            }
        } catch (error) {
            console.error(`Error fetching config for record ${recordId}:`, error);
            return null;
        }
    }

    async getInterpolatedHeartRate(recordId: string, frameStartTime: number, frameSize: number): Promise<InterpolatedData> {
        try {
            console.log(`Fetching interpolated data from /records-interpolated/${recordId}`);
            const response = await axios.get<InterpolatedData>(`${this.baseUrl}/records-interpolated/${recordId}`, {
                params: { frame_start_time: frameStartTime, frame_size: frameSize }
            });
            return {
                time: response.data?.time || [],
                value: response.data?.value || [],
                normalized: response.data?.normalized || [],
                mean: response.data?.mean || 0,
                std: response.data?.std || 0,
                dominant_freq_hz: response.data?.dominant_freq_hz || 0
            };
        } catch (error) {
            console.error(`Error fetching interpolated heart rate for record ${recordId}:`, error);
            
            // Create synthetic data points for the time range
            const timePoints = 500;
            const timeValues = Array.from({ length: timePoints }, 
                (_, i) => frameStartTime + (i * (frameSize / (timePoints - 1))));
            
            // Return placeholder data with time values but no heart rate values
            return {
                time: timeValues,
                value: timeValues.map(() => null),
                normalized: timeValues.map(() => null),
                mean: 0,
                std: 0,
                dominant_freq_hz: 0
            };
        }
    }

    async getHeartRateAnnotations(recordId: string): Promise<AnnotationsResponse> {
         try {
            console.log(`Fetching annotations from /records-annotations/${recordId}`);
            const response = await axios.get<AnnotationsResponse>(`${this.baseUrl}/records-annotations/${recordId}`);
            return { annotations: Array.isArray(response.data?.annotations) ? response.data.annotations : [] };
         } catch (error: any) {
             if (error.response && error.response.status === 404) {
                 console.log(`No annotations file found for record ${recordId}, returning empty.`);
                 return { annotations: [] };
             }
             console.error(`Error fetching annotations for record ${recordId}:`, error);
             return { annotations: [] };
         }
    }

    async saveHeartRateAnnotations(recordId: string, annotations: Annotation[]): Promise<boolean> {
        try {
            console.log(`Saving annotations to /records-annotations/${recordId}`);
            const response = await axios.post(`${this.baseUrl}/records-annotations/${recordId}`, { annotations });
            console.log(`Annotations saved for record ${recordId}`);
            return response.status >= 200 && response.status < 300;
        } catch (error) {
            console.error(`Error saving annotations for record ${recordId}:`, error);
            return false;
        }
    }

    async autoAnnotateHeartRate(recordId: string, frameStartTime: number, frameSize: number): Promise<AnnotationsResponse> {
        try {
            console.log(`Auto-annotating from /records-auto-annotate/${recordId}`);
            const response = await axios.post<AnnotationsResponse>(`${this.baseUrl}/records-auto-annotate/${recordId}`, {
                frame_start_time: frameStartTime,
                frame_size: frameSize
            });
            return { annotations: Array.isArray(response.data?.annotations) ? response.data.annotations : [] };
        } catch(error) {
            console.error(`Error auto-annotating heart rate for record ${recordId}:`, error);
            return { annotations: [] };
        }
    }
}
