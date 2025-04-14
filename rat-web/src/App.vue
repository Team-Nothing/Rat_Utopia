<template>
  <div class="app-container">
    <nav class="main-nav">
      <router-link to="/" class="nav-link">Home</router-link>
      <router-link to="/record" class="nav-link">Record</router-link>
      <router-link to="/breathe" class="nav-link">Breathe</router-link>
      <router-link to="/heart-rate" class="nav-link">Heart Rate</router-link>
    </nav>
    <div class="router-view-container">
      <RouterView />
    </div>
  </div>
</template>

<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import "@material/web/button/filled-button";
import "@material/web/button/filled-tonal-button";
import "@material/web/button/outlined-button";
import "@material/web/button/text-button";
import "@material/web/checkbox/checkbox";
import "@material/web/divider/divider";
import "@material/web/fab/fab";
import "@material/web/icon/icon";
import "@material/web/iconbutton/icon-button";
import "@material/web/iconbutton/filled-icon-button";
import "@material/web/iconbutton/filled-tonal-icon-button";
import "@material/web/iconbutton/outlined-icon-button";
import "@material/web/progress/linear-progress";
import "@material/web/progress/circular-progress";
import "@material/web/ripple/ripple";
import "@material/web/slider/slider";
import "@material/web/switch/switch";
import "@material/web/tabs/primary-tab";
</script>

<style lang="stylus">
.app-container
  display: flex
  flex-direction: column
  height: 100vh
  overflow: hidden

.main-nav
  display: flex
  background: var(--md-sys-color-primary)
  color: var(--md-sys-color-on-primary)
  height: 48px
  padding: 0 16px
  align-items: center
  position: fixed
  top: 0
  left: 0
  right: 0
  z-index: 1000
  
  .nav-link
    padding: 8px 16px
    color: var(--md-sys-color-on-primary)
    text-decoration: none
    border-radius: 4px
    margin-right: 8px
    
    &:hover
      background: rgba(255, 255, 255, 0.1)
    
    &.router-link-active
      background: var(--md-sys-color-primary-container)
      color: var(--md-sys-color-on-primary-container)

/* Add padding to RouterView to account for fixed navbar */
.app-container > .router-view-container
  padding-top: 48px
  height: calc(100vh - 48px)
  overflow: auto
</style>
