<template>
  <div class="combo-box">
    <md-outlined-button class="company-select" trailing-icon @click="showItems = true" ref="comboBoxRef">
      <div style="display: flex; align-items: center; justify-content: space-between">
        {{items[currIndex]}}
        <md-icon slot="icon">arrow_drop_down</md-icon>
      </div>
    </md-outlined-button>
    <div v-if="showItems" class="dropdown-list">
      <ul>
        <li v-for="(item, i) in items" :key="i" @click="handleSelectedChange(i)" >
          {{ item }}
        </li>
      </ul>
    </div>
  </div>
</template>

<script lang="ts">
import {computed, defineComponent, onMounted, onUnmounted, ref, watch} from "vue";
// import ColorIcon from "@/components/ColorIcon.vue";

export default defineComponent({
  name: 'ComboBox',
  components: {},
  props: {
    items: {
      type: Array,
      required: true,
    },
    default: {
      type: Number
    },
    onSelectedChange: Function
  },
  setup(props) {
    const currIndex = ref(props.default ?? 0)
    const showItems = ref(false)

    const comboBoxRef = ref<HTMLElement | null>(null)

    const handleClickOutside = (event: MouseEvent) => {
      if (comboBoxRef.value && !comboBoxRef.value.contains(event.target as Node)) {
        showItems.value = false
      }
    }

    const handleSelectedChange = (index: number) => {
      currIndex.value = index
      if (props.onSelectedChange) {
        props.onSelectedChange(index)
      }
    }

    onMounted(() => {
      document.addEventListener('click', handleClickOutside);
    });

    onUnmounted(() => {
      document.removeEventListener('click', handleClickOutside);
    });

    watch(
        () => props.default,
        (newValue) => {
          currIndex.value = newValue ?? 0; // Update the selected index
        }
    );

    return { currIndex, showItems, comboBoxRef, handleSelectedChange };
  }
})

</script>

<style lang="stylus" scoped>
.combo-box
  position relative
  .dropdown-list
    position: absolute
    background-color: var(--md-sys-color-on-secondary)
    top 0
    left 0
    width 100%
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4)
    border-radius: 16px
    z-index 10
    ul
      list-style-type: none
      padding: 8px
      margin: 0
      text-align start
    li
      font-size var(--md-sys-fontsize-title-medium)
      border-radius 12px
      padding: 12px 20px
      cursor: pointer
      &:hover
        background-color: var(--md-sys-color-tertiary-container)

</style>
