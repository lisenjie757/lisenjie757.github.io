import { navbar } from "vuepress-theme-hope";

// 精选图标：https://theme-hope.vuejs.press/zh/guide/interface/icon.html#iconfont-%E7%B2%BE%E9%80%89%E5%9B%BE%E6%A0%87
export default navbar([
  { text: "主页", icon: "home", link: "/"},
  { text: "博客", icon: "blog", link: "/blog" },
  {
    text: "知识库",
    icon: "study",
    link: "/知识库",
    prefix: "/知识库/",
  },

  { text: "资料收集", icon: "workingDirectory", link: "https://notes.orangex4.cool/?git=github&github=lisenjie757/blog" },
  { text: "工具收藏", icon: "tool", link: "https://nav.newzone.top/" },
]);
