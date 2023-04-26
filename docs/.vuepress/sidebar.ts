import { sidebar } from "vuepress-theme-hope";

// 精选图标：https://theme-hope.vuejs.press/zh/guide/interface/icon.html#iconfont-%E7%B2%BE%E9%80%89%E5%9B%BE%E6%A0%87
export default sidebar([
  // 读书笔记架构更换到 docsify，不能使用相对链接
  //{ text: "读书笔记", icon: "read", link: "https://lisenjie757.github.io/knowledge/reading/" },
  // 指定显示页面
  {
    text: "📑 文献精读",
    icon: "",
    prefix: "/知识库/文献精读/",
    link: "",
    collapsible: true,
    children: "structure",
  },
  {
    text: "💻 PyTorch入门",
    icon: "",
    prefix: "/知识库/PyTorch入门/",
    link: "",
    collapsible: true,
    children: "structure",
  },
]);
