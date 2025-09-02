# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Directories
plot_folder <- "/Users/jorisg/Desktop/upper_limb/paper_figures_revision-R"
data_folder <- "/Users/jorisg/Desktop/upper_limb/paper_data-2"

df <- read.csv(file.path(data_folder, 'realTime.csv'))
df$MSE <- df$Value
df$"Online Training Time [s]" <- df$X
df$dummy <- "Real-Time Performance"

df$Label <- factor(df$Label, levels = c("Regular", "Perturbed"))
df$Mode <- factor(df$Mode, levels = c("Offline Stream", "Interpolated Offline Stream", "Real-Time"))


# Create the plot
combined_plot <-ggplot(df, aes(x = factor(Mode), y = Value, fill = Label)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c("Regular" = "#53baa0", "Perturbed" = "#35618f")) +
  labs(x = "Training Data", y = "Total MSE") +
  
  facet_grid(
    cols = vars(dummy), 
    switch = "y"
  )  +
  guides(
    fill = guide_legend(title = "Data", ncol = 1, title.position = "top", title.hjust = 0.5)
  ) +
  theme(
        strip.placement = "outside",
    legend.justification = c("right", "top"), 
    legend.direction = "vertical", 
    legend.box = "horizontal", # Display color and linetype legends in one line
    legend.spacing = unit(20, "lines"), # Remove spacing between legend items
    legend.key.spacing.y = unit(0.7, "lines"), # Adjust vertical spacing between legend items
    # legend.key.height = unit(5, "lines"),
    legend.key = element_blank(),
    legend.position = c(0.98, 0.6),
    legend.title = element_text(size = 12, face = "bold"), 
    legend.background = element_rect(fill = "gray90"), # Sets background for entire legend area
    strip.text = element_text(size = 12, face = "bold"),
    axis.ticks.length = unit(0, "pt"), 
    panel.background = element_rect(fill = "gray98", color = NA), 
    panel.grid.minor = element_blank(), 
    panel.grid.major = element_line(size = 0.3, color = "gray90")
  ) 


ggsave(
  filename = file.path(plot_folder, "realTime.pdf"),
  plot = combined_plot,
  width = 9 * 0.7,
  height = 4 * 0.7,
  dpi = 600,
  device = "pdf"
)
