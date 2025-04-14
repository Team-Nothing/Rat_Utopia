import { createRouter, createWebHistory } from 'vue-router'
import RecordView from "@/views/RecordView.vue";
import HomeView from "@/views/HomeView.vue";
import BreatheView from "@/views/BreatheView.vue";
import HeartRateView from '../views/HeartRateView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    }, {
      path: '/record',
      name: 'record',
      component: RecordView,
    }, {
      path: '/breathe',
      name: 'breathe',
      component: BreatheView,
    }, {
      path: '/heart-rate',
      name: 'heart-rate',
      component: HeartRateView,
    },
  ],
})

export default router
