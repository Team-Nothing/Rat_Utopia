<template>
  <div class="main">
    <div class="device-preview">
      <div class="video-container">
        <img :src="cameraStream" alt="Camera Preview" />
      </div>
      <span>Camera</span>
      <div class="line-chart" ref="chartContainer">
        <line-chart
            :key="chartKey"
            :width="chartWidth"
            :height="chartHeight"
            :chart-data="chartData"
            :options="chartOptions"
        />
      </div>
      <span>Heart-Rate Sensor</span>
    </div>
    <div class="xdd">
      <div class="actions-container">
        <div class="actions">
          <div class="container primary">
            <span class="title">Record Actions</span>
            <div class="sub-container" style="gap: 16px">
              <div style="display: flex; flex-wrap: wrap">
                <span>Current Status</span>
                <span v-if="recordStatusRef">{{recordStatusRef.is_recording? ': RECORDING':': IDLE'}} {{recordStatusRef.recording_title}}</span>
                <span v-if="recordStatusRef && recordStatusRef.start_at">, start_at: {{recordStatusRef.start_at}}</span>
              </div>
              <div class="button-container">
                <combo-box
                    :items="Object.values(recordDurations)"
                    :default="durationRef"
                    :on-selected-change="durationOSC"/>
                <md-filled-icon-button :disabled="recordStatusRef?.is_recording" @click="startRecording()">
                  <md-icon>play_circle</md-icon>
                </md-filled-icon-button>
                <md-filled-icon-button style="--md-sys-color-primary: var(--md-sys-color-tertiary)" @click="stopRecording()"
                    :disabled="recordStatusRef?.recording_title || !recordStatusRef?.is_recording">
                  <md-icon>stop_circle</md-icon>
                </md-filled-icon-button>
                <md-filled-icon-button style="--md-sys-color-primary: var(--md-sys-color-error)" @click="stopRecording()"
                    :disabled="!recordStatusRef?.recording_title || !recordStatusRef?.is_recording">
                  <md-icon>report</md-icon>
                </md-filled-icon-button>
              </div>
            </div>
          </div>

          <div class="container secondary">
            <span class="title">Camera Settings</span>
            <div class="sub-container" style="gap: 8px">
              <div class="combo-container">
                <span>Camera:</span>
                <combo-box v-if="cameraConfigRef" :default="cameraRef.camera_index" :items="cameraConfigRef.CAMERAS"
                           :on-selected-change="cameraOSC"/>
                <md-circular-progress v-else class="indicator" indeterminate></md-circular-progress>
              </div>
              <div class="combo-container">
                <span>Resolution:</span>
                <combo-box v-if="cameraConfigRef" :default="cameraRef.camera_frame_index"
                           :items="cameraConfigRef.FRAMES.map((data) => `${data.width} x ${data.height}`)"
                           :on-selected-change="cameraResolutionOSC"/>
                <md-circular-progress v-else class="indicator" indeterminate></md-circular-progress>
              </div>
              <div class="combo-container">
                <span>FPS:</span>
                <combo-box v-if="cameraConfigRef" :default="cameraRef.camera_frame_rate_index"
                           :items="cameraConfigRef.FRAMES[cameraRef.camera_frame_index].rate"
                           :on-selected-change="cameraFramerateOSC"/>
                <md-circular-progress v-else class="indicator" indeterminate></md-circular-progress>
              </div>
              <div class="combo-container">
                <span>Light:</span>
                <combo-box v-if="cameraConfigRef" :default="cameraRef.light_index"
                           :items="cameraConfigRef.LIGHTS"
                           :on-selected-change="cameraLightOSC"/>
                <md-circular-progress v-else class="indicator" indeterminate></md-circular-progress>
              </div>
              <md-filled-button @click="updateCameraConfig()">
                UPDATE
              </md-filled-button>
            </div>
          </div>
          <div class="container secondary">
            <span class="title">Heart-Rate Sensor Settings</span>
            <div class="sub-container" style="gap: 8px">
              <div class="combo-container">
                <span>Sample Rate:</span>
                <combo-box v-if="heartRateConfigRef" :default="heartRateRef.sample_rate_index"
                           :items="heartRateConfigRef.SAMPLE_RATES" :on-selected-change="sampleRateOSC"/>
                <md-circular-progress v-else class="indicator" indeterminate></md-circular-progress>
              </div>
              <div class="combo-container">
                <span>Frame Size:</span>
                <combo-box v-if="heartRateConfigRef" :default="heartRateRef.frame_size_index"
                           :items="heartRateConfigRef.FRAME_SIZES" :on-selected-change="frameSizeOSC"/>
                <md-circular-progress v-else class="indicator" indeterminate></md-circular-progress>
              </div>
              <div class="combo-container">
                <span>Frame Delay</span>
                <combo-box v-if="heartRateConfigRef" :default="heartRateRef.frame_delay_index"
                           :items="heartRateConfigRef.FRAME_DELAYS" :on-selected-change="frameDelayOSC"/>
                <md-circular-progress v-else class="indicator" indeterminate></md-circular-progress>
              </div>
              <div class="combo-container">
                <span>Frame Reduce</span>
                <combo-box v-if="heartRateConfigRef" :default="heartRateRef.frame_reduce_index"
                           :items="heartRateConfigRef.FRAME_REDUCES" :on-selected-change="frameReduceOSC"/>
                <md-circular-progress v-else class="indicator" indeterminate></md-circular-progress>
              </div>
              <md-filled-button @click="updateHeartRateConfig()">
                UPDATE
              </md-filled-button>
            </div>
          </div>
          <div class="container secondary">
            <span class="title">Latency</span>
            <div class="sub-container" style="gap: 4px">
              <span>Server: {{latencyRef.server}}ms,</span>
              <span>Camera: cable (0ms),</span>
              <span>Heart-Rate Sensor: {{latencyRef.heartRate ?? -1}}ms</span>
            </div>
          </div>
          <div class="container secondary">
            <span class="title">Record Data</span>
            <div class="list">
              <button v-for="(item, i) in recordsRef" :key="i">{{item}}</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import {defineComponent, onMounted, onUnmounted, type Ref, ref} from "vue";
import { LineChart, useLineChart } from "vue-chart-3";
import { Chart, ChartData, ChartOptions, registerables } from "chart.js";
import ComboBox from "@/components/ComboBox.vue";
import RecordService from "@/service/record/RecordService";
import type {CameraConfig, HeartRateConfig, Latency, RecordStatus} from "@/service/record/Data";

Chart.register(...registerables);
Chart.defaults.backgroundColor = '#bec2ff';
Chart.defaults.borderColor = '#454556';
Chart.defaults.color = '#c5c5d9';

export default defineComponent({
  name: 'RecordView',
  components: {ComboBox, LineChart},
  setup() {
    const recordService = new RecordService("http://localhost:8080")

    const durationRef = ref(5)
    const recordDurations = {
      5: "5 seconds",
      10: "10 seconds",
      30: "30 seconds",
      60: "1 minute",
      180: "3 minutes",
      300: "5 minutes",
      600: "10 minutes",
      "-1": "continuously"
    }

    const recordsRef: Ref<Array<string> | null> = ref(null)
    recordService.getRecords().then(result => {
      recordsRef.value = result
    })
    recordService.onRecordsUpdate(result => {
      recordsRef.value = result
    })

    const recordStatusRef: Ref<RecordStatus | null> = ref(null)
    recordService.getRecordStatus().then(result => {
      recordStatusRef.value = result
    })
    recordService.onRecordStatusUpdate(result => {
      recordStatusRef.value = result
    })

    const cameraConfigRef: Ref<CameraConfig | null> = ref(null)
    const cameraRef = ref({
      camera_index: 0,
      camera_frame_rate_index: 0,
      camera_frame_index: 0,
      light_index: 0
    })
    const latencyRef: Ref<Latency> = ref({
      server: -1,
      heartRate: -1
    })

    const heartRateConfigRef: Ref<HeartRateConfig | null> = ref(null)
    const heartRateRef = ref({
      sample_rate_index: 0,
      frame_size_index: 0,
      frame_delay_index: 0,
      frame_reduce_index: 0
    })

    recordService.getCameraConfig().then(result => {
      cameraRef.value.camera_index = result.camera_index
      cameraRef.value.camera_frame_index = result.camera_frame_index
      cameraRef.value.camera_frame_rate_index = result.camera_frame_rate_index
      cameraRef.value.light_index = result.light_index
      cameraConfigRef.value = result
    })

    recordService.onCameraConfigUpdate(config => {
      cameraRef.value.camera_index = config.camera_index
      cameraRef.value.camera_frame_index = config.camera_frame_index
      cameraRef.value.camera_frame_rate_index = config.camera_frame_rate_index
      cameraRef.value.light_index = config.light_index
      cameraConfigRef.value = config
    })

    recordService.getHeartRateConfig().then(result => {
      heartRateRef.value.sample_rate_index = result.sample_rate_index
      heartRateRef.value.frame_size_index = result.frame_size_index
      heartRateRef.value.frame_delay_index = result.frame_delay_index
      heartRateRef.value.frame_reduce_index = result.frame_reduce_index
      heartRateConfigRef.value = result
    })

    recordService.onHeartRateData(data => {
      chartData.value.labels = data.map(data => data.T % 1000)
      chartData.value.datasets[0].data = data.map(data => data.p)
    })

    recordService.onHeartRateConfigUpdate(config => {
      heartRateRef.value.sample_rate_index = config.sample_rate_index
      heartRateRef.value.frame_size_index = config.frame_size_index
      heartRateRef.value.frame_delay_index = config.frame_delay_index
      heartRateRef.value.frame_reduce_index = config.frame_reduce_index
      heartRateConfigRef.value = config
    })

    setInterval(() => {
      recordService.checkLatency().then(result => {
        latencyRef.value = result
      })
    }, 3000)

    const chartContainer = ref<HTMLElement | null>(null)
    const chartWidth = ref("auto");
    const chartHeight = ref("auto");
    const chartKey = ref(0)
    const updateChartDimensions = () => {
      if (chartContainer.value) {
        chartWidth.value = `${chartContainer.value.offsetWidth}px`;
        chartHeight.value = `${chartContainer.value.offsetHeight}px`;
        chartKey.value += 1
      }
    };


    onMounted(() => {
      updateChartDimensions();
      // window.addEventListener("resize", updateChartDimensions);
    });

    onUnmounted(() => {
      // window.removeEventListener("resize", updateChartDimensions);
    });

    const heartRateSensorSetting = {
      sampleRate: ['1kHz', '100Hz', '10Hz', '1Hz']
    }
    const chartData = ref({
      labels: [],
      datasets: [
        {
          label: 'presure',
          data: [],
          borderColor: '#a7d394', // Line color
          backgroundColor: '#a7d394', // Background color
          fill: false,
          cubicInterpolationMode: 'monotone',
          tension: 0.4,
        }
      ]
    })
    const chartOptions = {
      responsive: false,
      plugins: {
        title: {
          display: false,
        },
        legend: {
          labels: {
            font: {
              size: 0
            }
          }
        }
      },
      interaction: {
        intersect: true,
      },
      scales: {
        x: {
          display: true,
          title: {
            display: false
          },
          ticks: {
            font: {
              size: 12, // Set the font size for the x-axis scale
            },
          },
        },
        y: {
          display: true,
          title: {
            display: false,
          },
          ticks: {
            font: {
              size: 12, // Set the font size for the y-axis scale
            },
          },
          // suggestedMin: 0,
          // suggestedMax: 50
        }
      },
      animation: false, // Disable animation
    };

    const cameraResolutionOSC = (value: number) => {
      cameraRef.value.camera_frame_index = value
      cameraRef.value.camera_frame_rate_index = 0
    }
    const cameraFramerateOSC = (value: number) => {
      cameraRef.value.camera_frame_rate_index= value
    }
    const cameraOSC = (value: number) => {
      cameraRef.value.camera_index = value
    }
    const cameraLightOSC = (value: number) => {
      cameraRef.value.light_index = value
    }

    const sampleRateOSC = (value: number) => {
      heartRateRef.value.sample_rate_index = value
    }

    const frameSizeOSC = (value: number) => {
      heartRateRef.value.frame_size_index = value
    }

    const frameDelayOSC = (value: number) => {
      heartRateRef.value.frame_delay_index = value
    }

    const frameReduceOSC = (value: number) => {
      heartRateRef.value.frame_reduce_index = value
    }

    const durationOSC = (value: number) => {
      durationRef.value = value
    }

    const updateHeartRateConfig = () => {
      recordService.updateHeartRateConfig(
          heartRateRef.value.sample_rate_index,
          heartRateRef.value.frame_size_index,
          heartRateRef.value.frame_delay_index,
          heartRateRef.value.frame_reduce_index,
      ).then(() => {
        console.log("success")
      })
    }

    const updateCameraConfig = () => {
      recordService.updateCameraConfig(
          cameraRef.value.camera_index,
          cameraRef.value.camera_frame_index,
          cameraRef.value.camera_frame_rate_index,
          cameraRef.value.light_index
      ).then(() => {
        console.log("success")
      })
    }

    const startRecording = () => {
      recordService.recordStart(+Object.keys(recordDurations)[durationRef.value]).then(() => {
        console.log("startRecording")
      })
    }

    const stopRecording = () => {
      recordService.recordStop().then(() => {
        console.log("stopRecording")
      })
    }

    return {
      recordsRef,
      recordDurations,
      durationRef,
      cameraConfigRef,
      cameraRef,
      heartRateConfigRef,
      heartRateRef,
      latencyRef,
      cameraStream: recordService.cameraStream(),
      chartData,
      chartOptions,
      chartWidth,
      chartHeight,
      chartContainer,
      chartKey,
      cameraResolutionOSC,
      cameraFramerateOSC,
      cameraOSC,
      cameraLightOSC,
      updateCameraConfig,
      updateHeartRateConfig,
      sampleRateOSC,
      frameSizeOSC,
      frameDelayOSC,
      frameReduceOSC,
      durationOSC,
      recordStatusRef,
      startRecording,
      stopRecording
    }
  }
})

</script>

<style lang="stylus" scoped>
.main
  width calc(100% - 48px)
  height calc(100vh - 48px)
  display grid
  grid-template-columns 1fr 1fr
  gap 24px
  padding 24px
  box-sizing border-box
  overflow hidden

  .xdd
    margin-block auto
    max-height 100%
    overflow hidden
    .actions-container
      width 100%
      height 100%
      display flex
      overflow-y auto
      align-items center

      .actions
        width 100%
        height fit-content
        display flex
        flex-wrap wrap
        text-align start
        gap 16px
        justify-content flex-start
        align-items stretch

        .container
          padding 24px
          height auto;
          border-radius 16px
          display flex
          flex-direction column
          gap 16px
          flex: 1 1 calc(50% - 24px) /* Take half the container width minus gap */
          box-sizing border-box
          .list
            max-height 200px
            display flex
            flex-direction column
            gap: 4px
            justify-content: start
            align-items start
            overflow-y auto
            button
              padding 4px
              border none
              background transparent
              color var(--md-sys-color-secondary)
          .sub-container
            height 100%
            display flex
            justify-content center
            flex-direction column
            .combo-container
              .indicator
                --md-circular-progress-size 24px
              align-items center
              display flex
              justify-content space-between
              gap 8px
            .button-container
              align-items center
              display flex
              gap 8px
          span
            white-space: nowrap
          .title
            font-weight 600
            font-size var(--md-sys-fontsize-title-large)
        .secondary
          background var(--md-sys-color-secondary-container)
          color var(--md-sys-color-on-secondary-container)
        .primary
          flex: 1 1 100%
          box-shadow: 4px 8px 16px rgba(128, 128, 128, 0.2)
          background var(--md-sys-color-primary-container)
          color var(--md-sys-color-on-primary-container)

  .device-preview
    display flex
    flex-direction column
    text-align center
    gap 8px
    justify-content center
    align-items center
    max-height 100%
    overflow-y auto

    .line-chart
      aspect-ratio 8 / 3.5
      width: 100%
      border-radius 16px
      height auto

    span
      font-weight 600
      margin 4px 0
      
    .video-container 
      width 100%
      display flex
      justify-content center
      align-items center
      
      img
        width: 100%
        aspect-ratio 8/3
        border-radius 16px
        object-fit contain

@media (max-width: 720px)
  .main
    grid-template-columns 1fr
    grid-template-rows 1fr 1fr

</style>
