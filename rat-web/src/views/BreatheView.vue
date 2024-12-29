<template>
  <div class="left-right">
    <div class="left">
      <div class="record-container">
        <span class="title">Record Data</span>
        <div class="list">
          <button v-for="(item, i) in records" :key="i" :class="recordSelected == i? 'selected': ''" @click="onRecordSelected(i)">
            {{item}}
            <md-ripple/>
          </button>
        </div>
      </div>
    </div>
    <div class="main">
      <div class="video-container" ref="videoContainer">
        <video
            ref="videoPlayer"
            width="100%"
            @timeupdate="handleTimeUpdate"
            @loadedmetadata="handleLoadedMetadata"
        >
          <source v-if="videoSrc" :src="videoSrc" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div ref="rectangle" class="rectangle" :style="rectangleStyle" @mousedown="startDragOrResize">
          <div class="resize-handle top-left" @mousedown.stop="startResize('top-left', $event)"/>
          <div class="resize-handle top-right" @mousedown.stop="startResize('top-right', $event)"/>
          <div class="resize-handle bottom-left" @mousedown.stop="startResize('bottom-left', $event)"/>
          <div class="resize-handle bottom-right" @mousedown.stop="startResize('bottom-right', $event)"/>
        </div>
      </div>

      <!-- Controls -->
      <div class="middle-action">
        <div class="container primary">
          <div class="action-btn">
            <md-filled-icon-button @click="playVideo" :disabled="isPlaying || !videoSrc">
              <md-icon>play_circle</md-icon>
            </md-filled-icon-button>
            <md-filled-icon-button
                style="--md-sys-color-primary: var(--md-sys-color-tertiary)"
                @click="pauseVideo" :disabled="!isPlaying">
              <md-icon>pause_circle</md-icon>
            </md-filled-icon-button>
          </div>
          <md-slider
              ref="slider"
              :value="currentTime"
              :min="0"
              :max="duration"
              step="0.01"
              @input="onSliderChange"/>
          <span>{{ formattedTime }}</span>
        </div>
        <div class="container secondary">
          <div class="sub-container">
            <span style="padding-inline: 16px">Frame-size:</span>
            <combo-box default="2" :items="['1 sec', '2 sec', '5 sec', '10sec', '20sec', '30sec']"></combo-box>
          </div>
          <div class="sub-container">
            <label class="check-container">
              <md-checkbox
                  checked
                  touch-target="wrapper"
                  @change="toggleDataset('r', $event)"
              ></md-checkbox>
              show-r
            </label>
            <label class="check-container">
              <md-checkbox
                  checked
                  touch-target="wrapper"
                  @change="toggleDataset('g', $event)"
              ></md-checkbox>
              show-g
            </label>
            <label class="check-container">
              <md-checkbox
                  checked
                  touch-target="wrapper"
                  @change="toggleDataset('b', $event)"
              ></md-checkbox>
              show-b
            </label>
            <label class="check-container">
              <md-checkbox
                  checked
                  touch-target="wrapper"
                  @change="toggleDataset('all', $event)"
              ></md-checkbox>
              show-all
            </label>
          </div>
        </div>
      </div>
      <div class="line-chart" ref="chartContainer">
        <line-chart
            :key="chartKey"
            :width="chartWidth"
            :height="chartHeight"
            :chart-data="chartData"
            :options="chartOptions"
        />
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import {computed, defineComponent, ref, onMounted, type Ref} from "vue";
import { LineChart } from "vue-chart-3";
import { Chart, registerables } from "chart.js";
import ComboBox from "@/components/ComboBox.vue";
import RecordService from "@/service/record/RecordService";
import _default from "chart.js/dist/core/core.interaction";

Chart.register(...registerables);
Chart.defaults.backgroundColor = "#bec2ff";
Chart.defaults.borderColor = "#454556";
Chart.defaults.color = "#c5c5d9";

export default defineComponent({
  name: "BreatheView",
  components: {ComboBox, LineChart },
  setup() {

    const recordService = new RecordService("http://localhost:8000")

    const records: Ref<Array<string>|null> = ref(null)
    const recordSelected = ref(-1)
    const videoSrc = ref(null)
    const videoPlayer = ref<HTMLVideoElement | null>(null);
    const currentTime = ref(0); // Current playback time in seconds
    const duration = ref(0); // Total duration of the video
    const isPlaying = ref(false)
    const slider = ref<HTMLElement | null>(null);

    // Computed property for formatted time (mm:ss:ms)
    const formattedTime = computed(() => {
      const totalMilliseconds = currentTime.value * 1000; // Convert seconds to milliseconds
      const minutes = Math.floor(totalMilliseconds / 60000);
      const seconds = Math.floor((totalMilliseconds % 60000) / 1000);
      const milliseconds = Math.floor(totalMilliseconds % 1000);

      return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(
          2,
          "0"
      )}:${String(milliseconds).padStart(3, "0")}`;
    });

    // Play the video
    const playVideo = () => {
      videoPlayer.value?.play();
      isPlaying.value = true;
    };

    // Pause the video
    const pauseVideo = () => {
      videoPlayer.value?.pause();
      isPlaying.value = false;
    };

    // Update currentTime when the slider value changes
    const onSliderChange = () => {
      const sliderValue = slider.value?.value; // Access the slider's value directly
      if (sliderValue) {
        currentTime.value = sliderValue;
        videoPlayer.value.currentTime = sliderValue
        sendChartConfig()
      }
    };

    // Sync slider with the video's current playback time
    const handleTimeUpdate = (event) => {
      if (videoPlayer.value) {
        currentTime.value = event.target.currentTime;
      }
    };

    // Set total duration when metadata is loaded
    const handleLoadedMetadata = () => {
      if (videoPlayer.value) {
        duration.value = videoPlayer.value.duration;
      }
    };

    recordService.getRecords().then((result) => {
      records.value = result
    })
    recordService.onRecordsUpdate(result => {
      records.value = result
    })

    const chartContainer = ref(null);
    const chartHeight = ref("auto");
    const chartWidth = ref("auto");
    const chartKey = ref(0);

    const chartSocket = ref<WebSocket | null>(null); // WebSocket reference
    const chartLabels = ref([])

    const onRecordSelected = (index) => {
      recordSelected.value = index;
      videoSrc.value = "http://localhost:8000/breathe/video-stream/" + records.value[index]
      isPlaying.value = false;
      if (videoPlayer.value) {
        videoPlayer.value.load();

        if (chartSocket.value) {
          chartSocket.value.close()
        }

        setTimeout(() => {
          const recordId = records.value[index];
          chartSocket.value = new WebSocket(`ws://localhost:8000/breathe/square-rgb-steam/${recordId}`);
          chartSocket.value.onopen = () => {
            console.log("WebSocket connected to square_rgb_steam endpoint");
            sendChartConfig()
          };

          chartSocket.value.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
              console.error("WebSocket error:", data.error);
              return;
            }

            chartLabels.value = data.time
            datasets.value.r.data = data.mean_r;
            datasets.value.g.data = data.mean_g;
            datasets.value.b.data = data.mean_b;
            datasets.value.all.data = data.mean_all;

            sendChartConfig()
          };

          chartSocket.value.onclose = () => {
            console.log("WebSocket connection closed");
          };

          chartSocket.value.onerror = (error) => {
            console.error("WebSocket error:", error);
          };
        }, 500)
      }
    }

    const sendChartConfig = () => {
      if (chartSocket.value && chartSocket.value.readyState === WebSocket.OPEN) {
        const jsonData = {
          current_time: currentTime.value,
          frame_size: 5,
          rect_start: [normalizedRectangle.value.x0, normalizedRectangle.value.y0],
          rect_end: [normalizedRectangle.value.x1, normalizedRectangle.value.y1]
        };
        chartSocket.value.send(JSON.stringify(jsonData));
      }
    }

    // Toggle states for r, g, b, and all
    const toggleStates = ref({
      r: true,
      g: true,
      b: true,
      all: true,
    });

    // Original dataset
    const datasets = ref({
      r: {
        label: "r",
        data: [],
        borderColor: "rgb(255 218 214)",
        backgroundColor: "rgb(147 0 10)",
        fill: false,
        cubicInterpolationMode: "monotone",
        tension: 0.4,
      },
      g: {
        label: "g",
        data: [],
        borderColor: "rgb(194 239 174)",
        backgroundColor: "rgb(42 79 31)",
        fill: false,
        cubicInterpolationMode: "monotone",
        tension: 0.4,
      },
      b: {
        label: "b",
        data: [],
        borderColor: "rgb(160 207 209)",
        backgroundColor: "rgb(0 55 57)",
        fill: false,
        cubicInterpolationMode: "monotone",
        tension: 0.4,
      },
      all: {
        label: "all",
        data: [],
        borderColor: "rgb(127 127 127)",
        backgroundColor: "rgb(25 25 25)",
        fill: false,
        cubicInterpolationMode: "monotone",
        tension: 0.4,
      },
    });

    // Reactive chartData
    const chartData = computed(() => {
      setTimeout(() => {
        chartKey.value += 1
      }, 25)
      return {
        labels: chartLabels.value,
        datasets: Object.keys(datasets.value)
            .filter((key) => toggleStates.value[key])
            .map((key) => datasets.value[key]),
      }
    });

    // Chart options
    const chartOptions = {
      responsive: false,
      plugins: {
        title: {
          display: false,
        },
        legend: {
          labels: {
            font: {
              size: 0,
            },
          },
        },
      },
      interaction: {
        intersect: true,
      },
      scales: {
        x: {
          display: true,
          ticks: {
            font: {
              size: 12,
            },
          },
        },
        y: {
          display: true,
          ticks: {
            font: {
              size: 12,
            },
          },
          suggestedMin: 0,
          suggestedMax: 1,
        },
      },
      animation: false,
    };

    // Handle toggling datasets
    const toggleDataset = (key, event) => {
      toggleStates.value[key] = event.target.checked;
    };

    const updateChartDimensions = () => {
      if (chartContainer.value) {
        chartWidth.value = `${chartContainer.value.offsetWidth}px`;
        chartHeight.value = `${chartContainer.value.offsetHeight}px`;
        chartKey.value += 1; // Force re-render
      }
    };

    const videoContainer = ref<HTMLDivElement | null>(null);
    const rectangle = ref<HTMLDivElement | null>(null);

    const rectanglePosition = ref({ x0: 50, y0: 50, width: 100, height: 100 }); // Default rectangle position in pixels
    const normalizedRectangle = ref({ x0: 0, y0: 0, x1: 0, y1: 0 }); // Normalized coordinates

    const isDragging = ref(false);
    const isResizing = ref(false);
    const startPoint = ref({ x: 0, y: 0 });
    const dragOffset = ref({ x: 0, y: 0 });

    // Style for the rectangle
    const rectangleStyle = computed(() => ({
      left: `${rectanglePosition.value.x0}px`,
      top: `${rectanglePosition.value.y0}px`,
      width: `${rectanglePosition.value.width}px`,
      height: `${rectanglePosition.value.height}px`,
    }));

    // Start dragging or resizing the rectangle
    const startDragOrResize = (event: MouseEvent) => {
      const rect = rectangle.value?.getBoundingClientRect();
      const x = event.clientX - (rect?.left || 0);
      const y = event.clientY - (rect?.top || 0);

      if (x > rect!.width - 10 && y > rect!.height - 10) {
        // Bottom-right corner: Resize
        isResizing.value = true;
        startPoint.value = { x: event.clientX, y: event.clientY };
      } else {
        // Inside rectangle: Drag
        isDragging.value = true;
        dragOffset.value = {
          x: event.clientX - rectanglePosition.value.x0,
          y: event.clientY - rectanglePosition.value.y0,
        };
      }

      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", stopInteraction); // Ensure mouseup stops the interaction
    };

    // Handle mouse movement
    const onMouseMove = (event: MouseEvent) => {
      if (!videoContainer.value) return;

      const container = videoContainer.value.getBoundingClientRect();

      if (isDragging.value) {
        // Dragging logic
        rectanglePosition.value.x0 = Math.min(
            Math.max(event.clientX - dragOffset.value.x, 0),
            container.width - rectanglePosition.value.width
        );
        rectanglePosition.value.y0 = Math.min(
            Math.max(event.clientY - dragOffset.value.y, 0),
            container.height - rectanglePosition.value.height
        );
      } else if (isResizing.value) {
        // Resizing logic
        const newWidth = Math.max(
            Math.min(event.clientX - rectanglePosition.value.x0, container.width - rectanglePosition.value.x0),
            10 // Minimum width
        );
        const newHeight = Math.max(
            Math.min(event.clientY - rectanglePosition.value.y0, container.height - rectanglePosition.value.y0),
            10 // Minimum height
        );

        rectanglePosition.value.width = newWidth;
        rectanglePosition.value.height = newHeight;
      }

      // Update normalized coordinates
      updateNormalizedCoordinates();
    };

    // Stop dragging or resizing
    const stopInteraction = () => {
      isDragging.value = false;
      isResizing.value = false;
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", stopInteraction);
      sendChartConfig()
    };

    // Update normalized rectangle coordinates
    const updateNormalizedCoordinates = () => {
      if (!videoContainer.value) return;
      const { width, height } = videoContainer.value.getBoundingClientRect();

      normalizedRectangle.value = {
        x0: rectanglePosition.value.x0 / width,
        y0: rectanglePosition.value.y0 / height,
        x1: (rectanglePosition.value.x0 + rectanglePosition.value.width) / width,
        y1: (rectanglePosition.value.y0 + rectanglePosition.value.height) / height,
      };
    };

    // Adjust rectangle when the container is resized
    const adjustRectangleSize = () => {
      updateNormalizedCoordinates();
    };

    const startResize = (handle: string, event: MouseEvent) => {
      isResizing.value = true;
      startPoint.value = { x: event.clientX, y: event.clientY };

      const handleResizeCallback = (e: MouseEvent) => handleResize(e, handle);

      window.addEventListener("mousemove", handleResizeCallback);
      window.addEventListener("mouseup", () => {
        stopInteraction();
        window.removeEventListener("mousemove", handleResizeCallback); // Remove specific resize listener
      });
    };

    const handleResize = (event: MouseEvent, handle: string) => {
      const dx = event.clientX - startPoint.value.x;
      const dy = event.clientY - startPoint.value.y;

      const container = videoContainer.value?.getBoundingClientRect();
      if (!container) return;

      switch (handle) {
        case "top-left":
          rectanglePosition.value.x0 = Math.min(
              Math.max(rectanglePosition.value.x0 + dx, 0),
              rectanglePosition.value.x0 + rectanglePosition.value.width - 10
          );
          rectanglePosition.value.y0 = Math.min(
              Math.max(rectanglePosition.value.y0 + dy, 0),
              rectanglePosition.value.y0 + rectanglePosition.value.height - 10
          );
          rectanglePosition.value.width -= dx;
          rectanglePosition.value.height -= dy;
          break;

        case "top-right":
          rectanglePosition.value.width = Math.max(
              rectanglePosition.value.width + dx,
              10
          );
          rectanglePosition.value.y0 = Math.min(
              Math.max(rectanglePosition.value.y0 + dy, 0),
              rectanglePosition.value.y0 + rectanglePosition.value.height - 10
          );
          rectanglePosition.value.height -= dy;
          break;

        case "bottom-left":
          rectanglePosition.value.x0 = Math.min(
              Math.max(rectanglePosition.value.x0 + dx, 0),
              rectanglePosition.value.x0 + rectanglePosition.value.width - 10
          );
          rectanglePosition.value.width -= dx;
          rectanglePosition.value.height = Math.max(
              rectanglePosition.value.height + dy,
              10
          );
          break;

        case "bottom-right":
          rectanglePosition.value.width = Math.max(
              rectanglePosition.value.width + dx,
              10
          );
          rectanglePosition.value.height = Math.max(
              rectanglePosition.value.height + dy,
              10
          );
          break;
      }

      startPoint.value = { x: event.clientX, y: event.clientY };

      // Ensure rectangle stays within container bounds
      rectanglePosition.value.x0 = Math.min(
          Math.max(rectanglePosition.value.x0, 0),
          container.width - rectanglePosition.value.width
      );
      rectanglePosition.value.y0 = Math.min(
          Math.max(rectanglePosition.value.y0, 0),
          container.height - rectanglePosition.value.height
      );

      updateNormalizedCoordinates();
    };

    onMounted(() => {
      updateChartDimensions();
      window.addEventListener("resize", adjustRectangleSize);
      adjustRectangleSize();
    });

    return {
      chartOptions,
      chartData,
      chartContainer,
      chartWidth,
      chartHeight,
      chartKey,
      toggleDataset,
      records,
      recordSelected,
      onRecordSelected,
      videoSrc,
      videoPlayer,
      formattedTime,
      playVideo,
      pauseVideo,
      onSliderChange,
      handleLoadedMetadata,
      handleTimeUpdate,
      currentTime,
      duration,
      isPlaying,
      slider,
      videoContainer,
      rectangle,
      rectangleStyle,
      normalizedRectangle,
      startDragOrResize,
      startResize,
      handleResize
    };
  },
});
</script>

<style lang="stylus" scoped>
.left-right
  display flex
  grid-template-columns auto auto
  justify-content center

  .left
    height calc(100vh - 48px)
    display flex
    align-items center
    .record-container
      display flex
      flex-direction column;
      padding 16px
      background var(--md-sys-color-secondary-container)
      color var(--md-sys-color-on-secondary-container)
      border-radius 24px
      height 50vh
      gap 16px
      width 300px
      .list
        height 100%
        border 1px solid var(--md-sys-color-on-secondary-container)
        border-radius 16px
        display flex
        flex-direction column
        padding 8px
        gap 8px
        overflow-y scroll
        .selected
          background var(--md-sys-color-secondary)
          color var(--md-sys-color-on-secondary)
        :not(.selected)
          color var(--md-sys-color-on-secondary-container)
        button
          padding 8px
          border-radius 16px
          position relative;
          border none
          background transparent
          color var(--md-sys-color-on-secondary-container)
      .title
        font-weight 600
        margin-block 24px
        font-size var(--md-sys-fontsize-headline-medium)
  .main
    display grid
    width fit-content
    grid-template-rows 1fr auto 1fr

    flex-direction column
    height calc(100vh - 48px)
    padding 24px
    gap 16px
    justify-content center
    .video-container
      border 1px solid var(--md-sys-color-secondary)
      border-radius 16px
      aspect-ratio 8/3
      position: relative
      display flex
      .rectangle
        position: absolute
        border: 2px solid var(--md-sys-color-error)
        border-radius 8px
        background: rgba(147, 0, 10, 0.1)

        .resize-handle
          border-radius 3px
          position: absolute
          width: 8px
          height: 8px
          background: var(--md-sys-color-error-container)
          border: 1px solid var(--md-sys-color-on-error-container)
          cursor: pointer

        .resize-handle.top-left
          top: -5px
          left: -5px
          cursor: nwse-resize

        .resize-handle.top-right
          top: -5px
          right: -5px
          cursor: nesw-resize

        .resize-handle.bottom-left
          bottom: -5px
          left: -5px
          cursor: nesw-resize

        .resize-handle.bottom-right
          bottom: -5px
          right: -5px
          cursor: nwse-resize
      video
        display block
        border-radius 16px
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover
    .middle-action
      display flex
      flex-direction column
      gap 8px
      .primary
        background var(--md-sys-color-primary-container)
        color var(--md-sys-color-on-primary-container)
      .secondary
        background var(--md-sys-color-secondary-container)
        color var(--md-sys-color-on-secondary-container)
      .container
        width calc(100% - 32px)
        display flex
        padding 8px 16px
        border-radius 16px
        gap 4px
        align-items center
        justify-content space-between
        md-slider
          width 100%
        .action-btn
          display flex
          gap 4px
        .sub-container
          display flex
          align-items center
          gap 8px
          .check-container
            display flex
            align-items center

</style>
