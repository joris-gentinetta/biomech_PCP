# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Directories
plot_folder <- "/home/haptix/haptix/biomech_PCP/R_plots/plots"
data_folder <- "/home/haptix/haptix/biomech_PCP/paper_utils"

dir.create(plot_folder, recursive = TRUE, showWarnings = FALSE)
ptc = c('P_149', 'P_238', 'P_407', 'P_426', 'P_577', 'P_668', 'P_711', 'P_950', 'P7_453', 'P6_820')
ptcID = c('P_149' = 'P1', 'P_238' = 'P2', 'P_407' = 'P3', 'P_426' = 'P4', 'P_577' = 'P5', 'P_668' = 'P6', 'P_711' = 'P7', 'P_950' = 'P8', 'P7_453' = 'A1', 'P6_820' = 'A2')
ptcL = c('P_149' = 'solid', 'P_238' = 'solid', 'P_407' = 'solid', 'P_426' = 'solid', 'P_577' = 'solid', 'P_668' = 'solid', 'P_711' = 'solid', 'P_950' = 'solid', 'P7_453' = 'dashed', 'P6_820' = 'dashed')
# use these colors: colors <- c("#53baa0", "#f89231", "#63a122", "#8323dc", "#3ba7e5", "#d77cdd", "#35618f") + 3 more:
ptcC = c('P_149' = "#53baa0", 'P_238' = "#f89231", 'P_407' = "#63a122", 'P_426' = "#8323dc", 'P_577' = "#3ba7e5", 'P_668' = "#d77cdd", 'P_711' = "#35618f", 'P_950' = "#a0e85b", 'P7_453' = "#154975", 'P6_820' = "#70c6ca")

# Load data
df_before <- read.csv(file.path(data_folder, 'df_before.csv'), row.names = 1)
df_after <- read.csv(file.path(data_folder, 'df_after.csv'), row.names = 1)

# Participants and mode names
participants <- rownames(df_before)
modes <- c("Combined", "New.Movements", "Known.Movements")

# Combine data for plotting
plot_data <- data.frame()
for (participant in participants) {
  for (mode in modes) {
    plot_data <- rbind(plot_data, data.frame(
      Participant = participant,
      Mode = mode,
      Phase = c("Before", "After"),
      MSE = c(df_before[participant, mode], df_after[participant, mode])
    ))
  }
}

# Specify the order of the Phase variable
plot_data$Phase <- factor(plot_data$Phase, levels = c("Before", "After"))
plot_data$Participant <- factor(plot_data$Participant, levels = ptc)
plot_data$Mode <- factor(plot_data$Mode, levels = modes)

plot_data$Mode <- factor(plot_data$Mode, levels = modes,
                         labels = c("Combined", "New Movements", "Known Movements"))




combined_plot <- ggplot(plot_data, aes(x = Phase, y = MSE, color = Participant, group = Participant, linetype = Participant)) +
  geom_line() +
  facet_grid(
    cols = vars(Mode), 
    switch = "y"
  ) +
  scale_color_manual(
    values = ptcC, 
    labels = ptcID
  ) +
  scale_linetype_manual(
    values = ptcL, 
    labels = ptcID
  ) +  
  labs(x = "Online Training", y = "MSE") +
  theme(
    strip.text.y.left = element_text(angle = 0, face = "bold"),
    strip.placement = "outside",
    plot.title = element_text(face = "bold"),
    legend.position = c(0.98, 0.95), 
    legend.justification = c("right", "top"), 
    legend.direction = "vertical", 
    legend.key = element_blank(),
    legend.title = element_text(size = 12, face = "bold"), 
    legend.background = element_rect(fill = "gray90"), # Sets background for entire legend area
    strip.text = element_text(size = 12, face = "bold"),
    axis.ticks.length = unit(0, "pt"), 
    panel.background = element_rect(fill = "gray98", color = NA), 
    panel.grid.minor = element_blank(), 
    panel.grid.major = element_line(size = 0.3, color = "gray90")
  ) +
  guides(
    color = guide_legend(title = "Participants", ncol = 2, title.position = "top", title.hjust = 0.5),
    linetype = guide_legend(title = "Participants", ncol = 2, title.position = "top", title.hjust = 0.5)
  ) +
  scale_x_discrete(expand = c(0.05, 0.05))


ggsave(
  filename = file.path(plot_folder, "individualLines.pdf"),
  plot = combined_plot,
  width = 9 * 0.7,
  height = 4 * 0.7,
  dpi = 600,
  device = "pdf"
)
