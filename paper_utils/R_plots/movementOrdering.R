# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Directories
plot_folder <- "/Users/jorisg/Desktop/upper_limb/paper_figures_revision-R"
data_folder <- "/Users/jorisg/Desktop/upper_limb/paper_data-2"

df <- read.csv(file.path(data_folder, 'movementOrdering.csv'))
df$MSE <- df$Value
df$"Online Training Time [s]" <- df$X
df$dummy <- "Optimal Movement Ordering"
# colors <- c("#53baa0", "#f89231", "#63a122", "#8323dc", "#3ba7e5", "#d77cdd", "#35618f")
colors = c('#3ba7e5', '#63a122')
combined_plot <- ggplot(df, aes(x = `Online Training Time [s]`, y = MSE, color = Color, group = Label, linetype = Linestyle)) +
# scale_color_identity()+
  geom_line() +
  facet_grid(
    cols = vars(dummy), 
    switch = "y"
  ) +
  scale_color_manual(values = colors, labels = c("Known Movements", "New Movements")) +
    scale_linetype_manual(values = c("solid", "11"), labels = c("New First", "New Last")) +
  guides(
    color = guide_legend(title = "Test Set", ncol = 1, title.position = "top", title.hjust = 0.5),
    linetype = guide_legend(title = "Order", ncol = 1, title.position = "top", title.hjust = 0.5)
  ) +
      scale_x_continuous(
      breaks = seq(0, 34, by = 34 / 3) ,  # Specify breaks every 60 units
      labels = c(0, 120, 240, 360)           # Label these breaks as 0 to 6
    ) +  
    theme(
    strip.text.y.left = element_text(angle = 0, face = "bold"),
    strip.placement = "outside",
    plot.title = element_text(face = "bold"),
    legend.position = c(0.98, 0.95), 
    legend.justification = c("right", "top"), 
    legend.direction = "horizontal", 
          legend.box = "horizontal", # Display color and linetype legends in one line
      legend.spacing = unit(0, "lines"), # Remove spacing between legend items

    legend.key = element_blank(),
    legend.title = element_text(size = 12, face = "bold"), 
    legend.background = element_rect(fill = "gray90"), # Sets background for entire legend area
    strip.text = element_text(size = 12, face = "bold"),
    axis.ticks.length = unit(0, "pt"), 
    panel.background = element_rect(fill = "gray98", color = NA), 
    panel.grid.minor = element_blank(), 
    panel.grid.major = element_line(size = 0.3, color = "gray90")
  ) 


ggsave(
  filename = file.path(plot_folder, "movementOrdering.pdf"),
  plot = combined_plot,
  width = 9 * 0.7,
  height = 4 * 0.7,
  dpi = 600,
  device = "pdf"
)
