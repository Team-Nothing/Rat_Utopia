import { createRouter, createWebHistory } from 'vue-router'
import RecordView from "@/views/RecordView.vue";
import HomeView from "@/views/HomeView.vue";
import BreatheView from "@/views/BreatheView.vue";

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
    },
  ],
})

export default router
