import torch
import torch.nn as nn

# feature extractor
from diffusion_net.layers import DiffusionNet


class toLimitshape(nn.Module):
    """Project vertex wise features towards the spectral space.
    Then project towards the limit shape basis."""

    def __init__(self):
        super().__init__()

    def forward(self, x, phi_pinv_shape, Y_pinv_shape):
        # 1. to spectral space
        # 2. to limit shape basis

        x_emb = torch.matmul(Y_pinv_shape, torch.matmul(phi_pinv_shape, x))

        return x_emb


class toTemplateshape(nn.Module):
    """Project shape features represented in the limit shape basis towards the spectral representation of the templates mesh.
    Then project towards the template mesh."""

    def __init__(self):
        super().__init__()

    def forward(self, x_emb, Y_template, Phi_template):
        x = torch.matmul(Phi_template, torch.matmul(Y_template, x_emb))

        return x


class concatenateTemplateFeatures(nn.Module):
    """Project shape features represented in the limit shape basis towards the spectral representation of the templates mesh.
    Then project towards the template mesh."""

    def __init__(self, Feature_template):
        super().__init__()
        self.concatenate_features = False
        if Feature_template is not None:
            self.concatenate_features = True

    def forward(self, x, Features_template):
        if self.concatenate_features:
            x = torch.cat((x, Features_template), dim=-1)

        return x



# limit shape diffusion net autoencoder. only trainable part are the diffusion net blocks
class LS_DF_net(nn.Module):
    """
    Global model :
    - diffusion net as feature extractor
    - diffusion net in decoder
    - project to limit shape in encoder
    - project to template shape in decoder
    - V2V loss
    """

    def __init__(self, cfg):
        super().__init__()

        # feature extractor #
        with_grad = True

        self.encoder_diffnet = DiffusionNet(
            C_in=cfg["dataset"]["ndim"],
            C_out=cfg["diffnet"]["nfeature"],
            C_width=cfg["diffnet"]["k_eig_enc"],
            N_block=cfg["diffnet"]["N_block_enc"],
            mlp_hidden_dims=[128, 128] * cfg["diffnet"]["expand_internal"],
            dropout=cfg["diffnet"]["dropout"],
            with_gradient_features=with_grad,
            with_gradient_rotations=with_grad,
        )

        # projection to and from limit shape
        self.toLimitshape = toLimitshape()

        self.toTemplateshape = toTemplateshape()
        self.concatenateTemplateFeatures = concatenateTemplateFeatures(
            cfg["model"]["template_features"]
        )

        self.decoder_diffnet = DiffusionNet(
            C_in=cfg["diffnet"]["nfeature"] + cfg["model"]["template_features_dim"],
            C_out=cfg["dataset"]["ndim"],
            C_width=cfg["diffnet"]["k_eig_dec"],
            N_block=cfg["diffnet"]["N_block_dec"],
            mlp_hidden_dims=[128, 128] * cfg["diffnet"]["expand_internal"],
            dropout=cfg["diffnet"]["dropout"],
            with_gradient_features=with_grad,
            with_gradient_rotations=with_grad,
        )

    def encoder(
        self,
        verts,
        phi_pinv_shape,
        Y_pinv_shape,
        mass,
        LL,
        evals,
        evecs,
        gradX,
        gradY,
        faces,
    ):
        # get vertex-wise diffusionnet features
        features = self.encoder_diffnet(
            verts,
            mass,
            L=LL,
            evals=evals,
            evecs=evecs,
            gradX=gradX,
            gradY=gradY,
            faces=faces,
        )

        emb = self.toLimitshape(features, phi_pinv_shape, Y_pinv_shape)

        return emb

    def decoder(
        self,
        emb,
        Y_template,
        Phi_template,
        Features_template,
        T_mass,
        LL,
        evals,
        evecs,
        gradX,
        gradY,
        faces,
    ):

        T_features = self.toTemplateshape(emb, Y_template, Phi_template)
        T_features = self.concatenateTemplateFeatures(T_features, Features_template)

        # predict 3D coordinates
        verts_reconstruct = self.decoder_diffnet(
            T_features,
            T_mass,
            L=LL,
            evals=evals,
            evecs=evecs,
            gradX=gradX,
            gradY=gradY,
            faces=faces,
        )

        return verts_reconstruct

    def forward(self, batch):
        verts, phi_pinv_shape, Y_pinv_shape = (
            batch["shape"]["xyz"],
            batch["shape"]["phi_pinv"],
            batch["shape"]["Y_pinv"],
        )
        faces, mass, LL, evals, evecs, gradX, gradY = (
            batch["shape"]["faces"],
            batch["shape"]["mass"],
            batch["shape"]["L"],
            batch["shape"]["evals"],
            batch["shape"]["evecs"],
            batch["shape"]["gradX"],
            batch["shape"]["gradY"],
        )

        Y_template, Phi_template = (batch["template"]["Y"], batch["template"]["phi"])
        Features_template = batch["template"]["meshfeatures"]
        T_faces, T_mass, T_LL, T_evals, T_evecs, T_gradX, T_gradY = (
            batch["template"]["faces"],
            batch["template"]["mass"],
            batch["template"]["L"],
            batch["template"]["evals"],
            batch["template"]["evecs"],
            batch["template"]["gradX"],
            batch["template"]["gradY"],
        )

        # ## ENCODER

        emb = self.encoder(
            verts,
            phi_pinv_shape,
            Y_pinv_shape,
            mass,
            LL=LL,
            evals=evals,
            evecs=evecs,
            gradX=gradX,
            gradY=gradY,
            faces=faces,
        )

        # ## DECODER

        verts_reconstruct = self.decoder(
            emb,
            Y_template,
            Phi_template,
            Features_template,
            T_mass,
            LL=T_LL,
            evals=T_evals,
            evecs=T_evecs,
            gradX=T_gradX,
            gradY=T_gradY,
            faces=T_faces,
        )

        return verts_reconstruct

    def only_encoder(self, batch):
        verts, phi_pinv_shape, Y_pinv_shape = (
            batch["shape"]["xyz"],
            batch["shape"]["phi_pinv"],
            batch["shape"]["Y_pinv"],
        )
        faces, mass, LL, evals, evecs, gradX, gradY = (
            batch["shape"]["faces"],
            batch["shape"]["mass"],
            batch["shape"]["L"],
            batch["shape"]["evals"],
            batch["shape"]["evecs"],
            batch["shape"]["gradX"],
            batch["shape"]["gradY"],
        )

        # ## ENCODER

        emb = self.encoder(
            verts,
            phi_pinv_shape,
            Y_pinv_shape,
            mass,
            LL=LL,
            evals=evals,
            evecs=evecs,
            gradX=gradX,
            gradY=gradY,
            faces=faces,
        )

        return emb
    
    def only_decoder(self, batch, emb):
        
        Y_template, Phi_template = (batch["template"]["Y"], batch["template"]["phi"])
        Features_template = batch["template"]["meshfeatures"]
        T_faces, T_mass, T_LL, T_evals, T_evecs, T_gradX, T_gradY = (
            batch["template"]["faces"],
            batch["template"]["mass"],
            batch["template"]["L"],
            batch["template"]["evals"],
            batch["template"]["evecs"],
            batch["template"]["gradX"],
            batch["template"]["gradY"],
        )
        # ## DECODER

        verts_reconstruct = self.decoder(
            emb,
            Y_template,
            Phi_template,
            Features_template,
            T_mass,
            LL=T_LL,
            evals=T_evals,
            evecs=T_evecs,
            gradX=T_gradX,
            gradY=T_gradY,
            faces=T_faces,
        )

        return verts_reconstruct



class p2p_to_FM(nn.Module):
    """Get the FM from the provided p2p maps from shape to template shape."""

    def __init__(self):
        super().__init__()

    def forward(self, p2p_21, evects1, evects2):
        """
        p2p_21    : (n2,) vertex to vertex map from target to source.
                    For each vertex on the target shape, gives the index of the corresponding vertex on mesh 1.
                    Can also be presented as a (n2,n1) sparse matrix.
        eigvects1 : (n1,k1) eigenvectors on source mesh. Possibly subsampled on the first dimension.
        eigvects2 : (n2,k2) eigenvectors on target mesh. Possibly subsampled on the first dimension.
        """

        evects1_pb = evects1[p2p_21, :]

        return evects2.T @ evects1_pb
    

 
