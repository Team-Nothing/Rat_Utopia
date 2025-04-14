<script setup lang="ts">
import { ref, onMounted, watch, onBeforeUnmount } from 'vue';
import { Chart } from 'chart.js';
import type { ChartData, ChartOptions } from 'chart.js';

// Define props
const props = defineProps<{
  chartData: ChartData<'line'>,
  chartOptions: ChartOptions<'line'>
}>();

// Reference to the canvas element
const canvas = ref<HTMLCanvasElement | null>(null);
// Reference to the Chart instance
const chart = ref<Chart | null>(null);

// Function to create or update the chart
const renderChart = () => {
  if (!canvas.value) {
    console.warn('Canvas element not found');
    return;
  }
  
  // If chart exists, destroy it
  if (chart.value) {
    chart.value.destroy();
    chart.value = null;
  }

  // Create new chart
  const ctx = canvas.value.getContext('2d');
  if (ctx) {
    console.log('Creating chart with datasets:', props.chartData.datasets.length);
    
    // Log a sample of data to verify it's being passed correctly
    props.chartData.datasets.forEach((dataset, i) => {
      const dataPoints = dataset.data?.length || 0;
      console.log(`Dataset ${i}: ${dataset.label}, points=${dataPoints}`);
      
      if (dataPoints > 0) {
        // Log first, middle, and last value to verify data
        const firstVal = dataset.data?.[0];
        const midVal = dataset.data?.[Math.floor(dataPoints/2)];
        const lastVal = dataset.data?.[dataPoints-1];
        console.log(`  Sample values: [0]=${firstVal}, [${Math.floor(dataPoints/2)}]=${midVal}, [${dataPoints-1}]=${lastVal}`);
      }
    });
    
    chart.value = new Chart(ctx, {
      type: 'line',
      data: props.chartData,
      options: {
        ...props.chartOptions,
        responsive: true,
        maintainAspectRatio: false,
        elements: {
          line: {
            tension: 0.1  // Slight smoothing for heart rate data
          },
          point: {
            radius: 0,    // Hide points by default
            hoverRadius: 6 // Show on hover
          }
        }
      }
    });
  } else {
    console.error('Failed to get 2D context from canvas');
  }
};

// Watch for changes in props to update chart
watch(() => props.chartData, renderChart, { deep: true });
watch(() => props.chartOptions, renderChart, { deep: true });

// Initialize chart when component is mounted
onMounted(() => {
  // Small delay to ensure the DOM is fully ready
  setTimeout(() => {
    console.log('Component mounted, rendering chart');
    renderChart();
  }, 50);
});

// Clean up chart when component is unmounted
onBeforeUnmount(() => {
  if (chart.value) {
    console.log('Cleaning up chart instance');
    chart.value.destroy();
    chart.value = null;
  }
});
</script>

<template>
  <div class="chart-wrapper">
    <div class="chart-container">
      <canvas ref="canvas"></canvas>
    </div>
  </div>
</template>

<style lang="stylus" scoped>
.chart-wrapper
  width 100%
  height 100%
  padding 4px
  box-sizing border-box

.chart-container
  position relative
  width 100%
  height 100%
  background var(--md-sys-color-surface-container)
  border-radius 16px
  overflow hidden
  box-shadow 0 1px 2px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)
  transition box-shadow 0.3s ease, transform 0.2s ease
  display flex
  align-items center
  justify-content center
  
  &:hover
    box-shadow 0 2px 4px rgba(0, 0, 0, 0.12), 0 3px 6px rgba(0, 0, 0, 0.1)
    transform translateY(-1px)

  canvas
    width 100%
    height 100%
    border-radius 16px
    padding 8px
    box-sizing border-box
</style> 